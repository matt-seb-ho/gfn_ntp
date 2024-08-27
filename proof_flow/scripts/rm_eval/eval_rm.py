import argparse
import json
import random
import itertools
from typing import Callable, Optional
# import vllm
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

from proof_flow.src.utils import get_config, repo_root
from proof_flow.src.gfn_tuning.verifier import batch_completion_probabilities
from proof_flow.src.gfn_tuning.reward import rm_formatting_func

"""
given: 
- set of theorems and tactic proofs
- a set of generated proof completions for each theorem (STEP 1)
- verification results of proof completions (STEP 2)
    - for each theorem, for each completion...
      - whether the completion yields a correct proof
      - list[tactic_dict], where tactic_dict has form {state_before: str, tactic: str, state_after: str}
- (STEP 3) paired tactic data 
{
    "thm_idx": str,
    "state_before": str,
    "positive": list[TacticEntry]
    "negative": list[TacticEntry]
}
TacticEntry has shape:
{
    "tactic": str,
    "state_before": str,
    "state_after": str,
    "dist_reduction": float
}

goal:
- script that takes in a given model (and tokenizer) and evaluates its performance on the RM evaluation

current step (STEP 4):
- need to pick out actual tactics
- there are multiple +/- options for each state
- we can choose to evaluate over all pairings
- or just pick one for each state
"""

SANITY_CHECK = False


def evaluate_reward_model(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    pair_data: list[dict],
    formatting_func: Callable = rm_formatting_func,
    pair_selection_strategy: str = "first", # "first", "random", "all"
    use_next_state: bool = False,
    device: Optional[torch.device] = None,
) -> dict:
    # first select pairs from pair_data
    selected_pairs = select_pairs(pair_selection_strategy, pair_data)
    
    results = {"acc": None, "completion_log_prob": {}, "correct": 0, "total": None}
    # then evaluate the model on these pairs
    for pair_data in tqdm(selected_pairs):
        pos_neg_log_probs = {}
        for key in ["positive", "negative"]:
            prompt_completion_pair = [
                formatting_func(
                    pair_data["state_before"], 
                    pair_data[key], 
                    next_state=(pair_data[f"state_after_{key}"] if use_next_state else None),
                ),
            ]

            if SANITY_CHECK:
                print("sanity check")
                print(
                    prompt_completion_pair[0][0],
                    prompt_completion_pair[0][1],
                    sep="(<-prompt)(completion->)",
                )

            completion_probs = batch_completion_probabilities(
                model, 
                tokenizer, 
                prompt_completion_pair,
                device=device,
            )
            log_prob = completion_probs[0]["log_prob_sum"]
            pos_neg_log_probs[key] = log_prob

        results["completion_log_prob"][pair_data["thm_idx"]] = pos_neg_log_probs
        if pos_neg_log_probs["positive"] > pos_neg_log_probs["negative"]:
            results["correct"] += 1

    results["total"] = len(selected_pairs)
    results["acc"] = results["correct"] / len(selected_pairs)
    return results
    

def select_pairs(
    strategy: str | int,
    pair_data: list[dict],
    random_seed: int = 42,
    n: int = 1,
) -> list[dict]:
    selected = []
    if strategy == "first":
        for entry in pair_data:
            selected.append(_pair_info(entry, 0, 0))
    elif strategy == "random":
        random.seed(random_seed)
        for entry in pair_data:
            if n == 1:
                positive_idx = random.randint(0, len(entry["positive"]) - 1)
                negative_idx = random.randint(0, len(entry["negative"]) - 1)
                selected.append(_pair_info(entry, positive_idx, negative_idx))
            else:
                # select n random pairs from cartesian product
                all_pairs = list(itertools.product(range(len(entry["positive"])), range(len(entry["negative"]))))
                if len(all_pairs) < n:
                    selected_pair_idxs = all_pairs
                else:
                    selected_pair_idxs = random.sample(all_pairs, n)
                for positive_idx, negative_idx in selected_pair_idxs:
                    selected.append(_pair_info(entry, positive_idx, negative_idx))
    elif strategy == "all":
        for entry in pair_data:
            iterator = itertools.product(range(len(entry["positive"])), range(len(entry["negative"])))
            for positive_idx, negative_idx in iterator:
                selected.append(_pair_info(entry, positive_idx, negative_idx))
    else:
        raise ValueError(f"Invalid pair selection strategy: {strategy}")
    return selected
    

def _pair_info(full_entry, positive_idx, negative_idx):
    return {
        "thm_idx": full_entry["thm_idx"],
        "state_before": full_entry["state_before"],
        "positive": full_entry["positive"][positive_idx]["tactic"],
        "negative": full_entry["negative"][negative_idx]["tactic"],
        "state_after_positive": full_entry["positive"][positive_idx]["state_after"],
        "state_after_negative": full_entry["negative"][negative_idx]["state_after"],
    }


if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument("--model", type=str)
    args = psr.parse_args()
    overrides = None
    if args.model:
        overrides = f"model={args.model}"
    cfg = get_config(config_name="rm_eval", overrides=overrides)

    model_id = cfg.model
    device = torch.device("cuda")

    if cfg.use_peft:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id, 
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
        )

    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(repo_root() / cfg.rm_data_file) as f:
        pair_data = json.load(f)
    
    if SANITY_CHECK:
        pair_data = pair_data[:2]
    
    results = evaluate_reward_model(model, tokenizer, pair_data, device=device)
    
    model_name = model_id.split("/")[-1]
    filename = f"{model_name}_rm_eval.json"
    with open(repo_root() / cfg.rm_eval_results_dir / filename, "w") as f:
        json.dump(results, f, indent=2)
    