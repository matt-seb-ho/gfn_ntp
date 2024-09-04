from collections import defaultdict
from typing import Optional
from contextlib import contextmanager

import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer
)

from proof_flow.src.constants import (
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
    device: Optional[str | torch.device] = None,
    length_penalty: bool = True,
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
    is_correct = torch.tensor(
        [1 if t_states[-1] == PROOF_COMPLETE_MESSAGE else 0 for t_states in states],
        device=device,
    )
    
    # - model based score
    idxs = []
    prompt_completion_pairs = []
    for trajectory_idx, (_states, _tactics) in enumerate(zip(states, tactics)):
        # _states: list[str]: represents states for this trajectory
        # _tactics: list[str]: represents tactics for this trajectory
        # for state_i, (state, tactic) in enumerate(zip(_states, _tactics), start=1):
        for idx in range(len(_tactics)):
            rm_inputs = build_reward_inputs(
                _states[idx], 
                _tactics[idx], 
                _states[idx + 1],
                use_sts_format=use_sts_format,
                prompts_for_model=prompts_for_model,
            )
            idxs.append(trajectory_idx)
            prompt_completion_pairs.append(rm_inputs)

    stepwise_scores = []
    for batch in batch_iterator(prompt_completion_pairs, batch_size):
        log_ps, lengths = batch_completion_probabilities(
            model, 
            tokenizer, 
            batch,
            device=device,
        )
        if length_penalty:
            stepwise_scores.append(log_ps / lengths)
        else:
            stepwise_scores.append(log_ps)
        
    idxs = torch.tensor(idxs, device=device)
    stepwise_scores = torch.cat(stepwise_scores)
    model_scores = torch.zeros(
        len(prompt_completion_pairs), 
        dtype=stepwise_scores.dtype,
        device=device
    )
    model_scores = model_scores.scatter_add(0, idxs, stepwise_scores)

    return torch.log(
        torch.max(
            torch.stack(is_correct - 1, model_scores, dim=-1), 
            dim=-1,
        )
    )


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
        verifier_adapter_name: Optional[str] = None,
    ):
        self.model = model,
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.verifier_batch_size = verifier_batch_size
        self.verifier_adapter_name = verifier_adapter_name

    
    def score(
        self,
        states: list[list[str]],
        tactics: list[list[str]],
        batch_size: Optional[int] = None,
        device: Optional[str | torch.device] = None,
    ) -> torch.Tensor:
        # prep batch_size
        batch_size = (
            batch_size 
            or self.verifier_batch_size 
            or DEFAULT_VERIFIER_BATCH_SIZE
        )
        with self._compute_reward_ctx():
            log_reward = compute_log_reward(
                states, 
                tactics, 
                self.model, 
                self.tokenizer, 
                batch_size=batch_size,
                device=device,
            )
        return log_reward
    
    
    @contextmanager
    def _compute_reward_ctx(self):
        """
        context manager for reward computation

        responsible for ensuring that during reward computation:
        - verifier adapter is active (or no adapter is active)
        - model is in eval mode
        - gradients are not computed
        and of course, restoring the model to its previous state afterwards
        """
        was_training: bool = self.model.training
        previously_active_adapter: str = self.model.active_adapters[0]

        if self.verifier_adapter_name is None:
            # verifier/RM does *not* have an adapter
            # - disable current [policy] adapter
            self.model.eval()
            with self.model.disable_adapter(), torch.no_grad():
                yield
        else:
            # verifier/RM has an adapter
            # - swap to verifier adapter and swap back after
            self.model.set_adapter(self.verifier_adapter_name)
            self.model.eval()
            with torch.no_grad():
                yield
            self.model.set_adapter(previously_active_adapter)
        
        # restore training mode if necessary
        if was_training:
            self.model.train()
