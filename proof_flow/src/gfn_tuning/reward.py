import gzip
import heapq
import pickle
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer
)

from proof_flow.src.constants import (
    DEFAULT_VERIFIER_ADAPTER_NAME,
    DEFAULT_VERIFIER_BATCH_SIZE,
    PROOF_COMPLETE_MESSAGE
)
from proof_flow.src.gfn_tuning.verifier import (
    batch_completion_probabilities,
    batch_iterator
)
from proof_flow.src.prompts import (
    INSTRUCTION_PROMPT_TEMPLATE,
    INSTRUCTION_COMPLETION_TEMPLATE,
    INSTRUCTION_COMPLETION_TEMPLATE_WITH_NEXT_STATE,
    RM_TEMPLATES,
)


def lora_to_base(model):
    model.base_model.disable_adapter_layers()
    model.eval()


def base_to_lora(model):
    model.base_model.enable_adapter_layers()
    model.train()


def build_reward_inputs(
    state: str,
    tactic: str,
    next_state: Optional[str] = None,
    use_sts_format: bool = False,
    prompts_for_model: Optional[str] = "llemma",
) -> tuple[str, str]:
    st_or_sts = "sts" if use_sts_format else "st"
    templates = RM_TEMPLATES[prompts_for_model][st_or_sts]
    return (
        templates["prompt"].format(
            state=state, tactic=tactic, next_state=next_state
        ),
        templates["completion"].format(
            state=state, tactic=tactic, next_state=next_state
        ),
    )


def compute_log_reward(
    states: list[list[str]],
    tactics: list[list[str]],
    model: PeftModel,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    use_sts_format: bool = False,
    prompts_for_model: Optional[str] = "llemma",
) -> torch.Tensor:
    """
    Computes reward for a batch of trajectores (states, tactics) using heuristics and model.
    
    Arguments
        states: 2D list of shape (batch_size, trajectory_length) containing states
        tactics: 2D list of shape (batch_size, trajectory_length - 1) containing tactics
        model: verifier for scoring the generated text
        tokenizer: AutoTokenizer for encoding the input
    """
    assert len(states) == len(tactics)
    # r = max(is_correct - 1, model_score)
    # - heuristic score
    is_correct = torch.tensor([[1 if state[-1] == PROOF_COMPLETE_MESSAGE else 0 for state in states]])
    
    # - model based score
    prompt_completion_pairs = []
    batch_idx_to_pair_idx = defaultdict(list)
    for batch_i, (_states, _tactics) in enumerate(zip(states, tactics)):
        # _states: list[str]: represents states for this trajectory
        # _tactics: list[str]: represents tactics for this trajectory
        # for state_i, (state, tactic) in enumerate(zip(_states, _tactics), start=1):
        for idx in range(len(_tactics)):
            batch_idx_to_pair_idx[batch_i].append(len(prompt_completion_pairs))
            rm_inputs = build_reward_inputs(
                _states[idx], 
                _tactics[idx], 
                _states[idx + 1],
                use_sts_format=use_sts_format,
                prompts_for_model=prompts_for_model,
            )
            prompt_completion_pairs.append(rm_inputs)
    results = []
    for batch in batch_iterator(prompt_completion_pairs, batch_size):
        results.extend(batch_completion_probabilities(model, tokenizer, batch))
    model_scores = torch.tensor([
        sum(results[j]["log_prob_sum"] for j in batch_idx_to_pair_idx[i])
        for i in range(len(states))
    ])

    return torch.log(torch.max(torch.stack(is_correct - 1, model_scores, dim=-1), dim=-1))


def rm_formatting_func(
    state: str, 
    tactic: str, 
    next_state: Optional[str]
) -> tuple[str, str]:
    prompt = INSTRUCTION_PROMPT_TEMPLATE.format(state=state)
    if next_state is None:
        completion = INSTRUCTION_COMPLETION_TEMPLATE.format(tactic=tactic)
    else:
        completion = INSTRUCTION_COMPLETION_TEMPLATE_WITH_NEXT_STATE.format(
            tactic=tactic, 
            next_state=next_state
        )
    return prompt, completion

        
def format_prompt(initial_state: str, tactic: Optional[str], resulting_state: Optional[str]) -> str:
    # TODO: finalize this to match the verifier's training
    if tactic is not None:
        if resulting_state is None:
            return f"{initial_state}\n###\n{tactic}\n###\n{resulting_state}"
        return f"{initial_state}\n###\n{tactic}"
    return f"{initial_state}\n###\n"


class NTPReward:
    def __init__(
        self, 
        model: Optional[PeftModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        temperature: float = 1.0, 
        verifier_batch_size: Optional[int] = None,
        verifier_adapter_name: str = DEFAULT_VERIFIER_ADAPTER_NAME,
    ):
        self.temperature = temperature
        self.verifier_batch_size = verifier_batch_size
        self.verifier_adapter_name = verifier_adapter_name

        assert (model is None) == (tokenizer is None)
        self.model = model,
        self.tokenizer = tokenizer

    
    def score(
        self,
        states: list[list[str]],
        tactics: list[list[str]],
        batch_size: Optional[int] = None
    ):
        batch_size = batch_size or self.verifier_batch_size or DEFAULT_VERIFIER_BATCH_SIZE
        # swap to verification adapters
        training = self.model.training
        previous_adapter = self.model.active_adapter
        self.model.set_adapter(self.verifier_adapter_name)
        self.model.eval()
        with torch.no_grad():
            log_reward = compute_log_reward(
                states, 
                tactics, 
                self.model, 
                self.tokenizer, 
                batch_size=batch_size
            )
        # reset model to original state
        self.model.set_adapter(previous_adapter)
        if training:
            self.model.train()
        return log_reward
