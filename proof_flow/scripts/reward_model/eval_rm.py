import argparse
import json
import random
import itertools
from typing import Optional
# import vllm
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

from proof_flow.src.utils import (
    batch_completion_probabilities,
    get_config, 
    repo_root, 
    set_up_padding,
)
from proof_flow.src.gfn_tuning.reward import build_reward_inputs

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
    pair_selection_strategy: str = "first", # "first", "random", "all"
    max_pairs_per_state: int = 1,
    use_sts_format: bool = False,
    prompts_for_model: str = "llemma",
    device: Optional[torch.device] = None,
    normalize_length: bool = False,
) -> dict:
    # first select pairs from pair_data
    selected_pairs = select_pairs(
        pair_selection_strategy, 
        pair_data,
        max_pairs_per_state=max_pairs_per_state,
    )
    
    results = {
        "acc": None, 
        "correct": 0, 
        "total": None,
        "score_method": "log_prob_avg" if normalize_length else "log_prob_sum",
        "scores": {}, 
    }
    # then evaluate the model on these pairs
    for i, pair_data in tqdm(enumerate(selected_pairs), total=len(selected_pairs)):
        scores_entry = {
            "thm_idx": pair_data["thm_idx"],
            "state": pair_data["state_before"],
            "positive_tactic": pair_data["positive"],
            "negative_tactic": pair_data["negative"],
            "state_after_positive": pair_data["state_after_positive"],
            "state_after_negative": pair_data["state_after_negative"],
        }
        for key in ["positive", "negative"]:
            prompt_completion_pair = [
                build_reward_inputs(
                    pair_data["state_before"], 
                    pair_data[key], 
                    pair_data[f"state_after_{key}"],
                    use_sts_format=use_sts_format,
                    prompts_for_model=prompts_for_model,
                ),
            ]

            if SANITY_CHECK:
                print("sanity check")
                print(
                    prompt_completion_pair[0][0],
                    prompt_completion_pair[0][1],
                    sep="(<-prompt)(completion->)",
                )

            log_ps, completion_lengths = batch_completion_probabilities(
                model, 
                tokenizer, 
                prompt_completion_pair,
                device=device,
            )

            score = log_ps[0].item()
            if normalize_length:
                score = score / completion_lengths[0].item()
            scores_entry[key] = score

        results["scores"][i] = scores_entry
        if scores_entry["positive"] > scores_entry["negative"]:
            results["correct"] += 1

    results["total"] = len(selected_pairs)
    results["acc"] = results["correct"] / len(selected_pairs)
    return results
    

def select_pairs(
    strategy: str | int,
    pair_data: list[dict],
    random_seed: int = 42,
    max_pairs_per_state: int = 1,
) -> list[dict]:
    selected = []
    if strategy == "first":
        for entry in pair_data:
            selected.append(_pair_info(entry, 0, 0))
    elif strategy == "random":
        random.seed(random_seed)
        for entry in pair_data:
            if max_pairs_per_state == 1:
                positive_idx = random.randint(0, len(entry["positive"]) - 1)
                negative_idx = random.randint(0, len(entry["negative"]) - 1)
                selected.append(_pair_info(entry, positive_idx, negative_idx))
            else:
                # select n random pairs from cartesian product
                all_pairs = list(itertools.product(range(len(entry["positive"])), range(len(entry["negative"]))))
                if len(all_pairs) < max_pairs_per_state:
                    selected_pair_idxs = all_pairs
                else:
                    selected_pair_idxs = random.sample(all_pairs, max_pairs_per_state)
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
    psr.add_argument("--suffix", type=str)
    psr.add_argument("--cfg", type=str)
    args = psr.parse_args()
    overrides = None
    if args.model:
        overrides = f"model={args.model}"
    config_name = args.cfg or "rm_eval"
    cfg = get_config(config_name=config_name, overrides=overrides)

    model_id = cfg.model
    device = torch.device("cuda")

        
    if cfg.use_peft:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    set_up_padding(model, tokenizer)
    model.to(device)

    with open(repo_root() / cfg.rm_data_file) as f:
        pair_data = json.load(f)
    
    if SANITY_CHECK:
        pair_data = pair_data[:2]
    
    results = evaluate_reward_model(
        model, 
        tokenizer, 
        pair_data, 
        device=device,
        pair_selection_strategy=cfg.pair_selection_strategy,
        max_pairs_per_state=cfg.max_pairs_per_state,
        use_sts_format=cfg.use_sts_format,
        prompts_for_model=cfg.prompts_for_model,
        normalize_length=cfg.normalize_length,
    )
    
    model_name = model_id.split("/")[-1]
    if cfg.rm_eval_result_filename is not None:
        filename = cfg.rm_eval_result_filename
    elif args.suffix:
        filename = f"{model_name}_rm_eval_{args.suffix}.json"
    else:
        filename = f"{model_name}_rm_eval.json"
    with open(repo_root() / cfg.rm_eval_results_dir / filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Accuracy: {results['acc']}")
