from collections import defaultdict
from typing import Optional
from contextlib import contextmanager

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from proof_flow.src.constants import (
    DEFAULT_VERIFIER_BATCH_SIZE,
    PROOF_COMPLETE_MESSAGE,
    TACTIC_ERROR_STRINGS
)
from proof_flow.src.utils import (
    batch_completion_probabilities,
    batch_iterator
)
from proof_flow.src.prompts import RM_TEMPLATES


BINARY_REWARD_VALUE = -100


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


class NTPReward:
    def __init__(
        self, 
        model: PeftModel,
        tokenizer: AutoTokenizer,
        temperature: float = 1.0, 
        verifier_batch_size: Optional[int] = None,
        verifier_adapter_name: Optional[str] = None,
    ):
        self.model = model
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
            # log_reward = self.compute_log_reward(
            log_reward = self.compute_binary_log_reward(
                states, 
                tactics, 
                self.model, 
                self.tokenizer, 
                batch_size=batch_size,
                device=device,
            )
        return log_reward
    
    
    def compute_log_reward(
        self,
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

        New Formulation (Pseudo-code):
        if trajectory_correct:
            return 0
        elif trajectory has bad tactic:
            return -100
        else:
            return max(reward_model_scores, -100)
        
        Arguments
            states: 2D list of shape (batch_size, trajectory_length) containing states
            tactics: 2D list of shape (batch_size, trajectory_length - 1) containing tactics
            model: verifier for scoring the generated text
            tokenizer: AutoTokenizer for encoding the input
        """
        assert len(states) == len(tactics) # batch_size
        
        # for each trajectory
        # - either assign heuristic score or queue prompt-completion job
        log_r = torch.zeros(len(states), device=device)
        trajectory_groups = []
        prompt_completion_pairs = []
        for i, (_states, _tactics) in enumerate(zip(states, tactics)):
            # _states: list[str]: represents states for this trajectory
            # _tactics: list[str]: represents tactics for this trajectory
            if _states[-1] == PROOF_COMPLETE_MESSAGE:
                # log_r[i] = 0
                continue
            elif self._is_tactic_result_an_error(_states[-1]):
                log_r[i] = -100
            else:
                # log_r[i] = -100
                # queue prompt-completion logp jobs
                for step_idx in range(len(_tactics)):
                    rm_inputs = build_reward_inputs(
                        _states[step_idx], 
                        _tactics[step_idx], 
                        _states[step_idx + 1],
                        use_sts_format=use_sts_format,
                        prompts_for_model=prompts_for_model,
                    )
                    trajectory_groups.append(i)
                    prompt_completion_pairs.append(rm_inputs)
                
        # run queued prompt-completion jobs
        if prompt_completion_pairs:
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
                
            stepwise_scores = torch.cat(stepwise_scores)
            log_r = log_r.scatter_add(
                0,                                               # dim
                torch.tensor(trajectory_groups, device=device),  # indices
                stepwise_scores                                  # values
            )

        # clip reward to -100
        log_r = torch.clamp(log_r, min=-100)
        return log_r
    

    def compute_binary_log_reward(
        self,
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

        New Formulation (Pseudo-code):
        if trajectory_correct:
            return 0
        elif trajectory has bad tactic:
            return -100
        else:
            return max(reward_model_scores, -100)
        
        Arguments
            states: 2D list of shape (batch_size, trajectory_length) containing states
            tactics: 2D list of shape (batch_size, trajectory_length - 1) containing tactics
            model: verifier for scoring the generated text
            tokenizer: AutoTokenizer for encoding the input
        """
        assert len(states) == len(tactics) # batch_size
        
        log_r = torch.zeros(len(states), device=device)
        for i, (_states, _tactics) in enumerate(zip(states, tactics)):
            # _states: list[str]: represents states for this trajectory
            # _tactics: list[str]: represents tactics for this trajectory
            if _states[-1] == PROOF_COMPLETE_MESSAGE:
                # log_r[i] = 0
                continue
            else:
                log_r[i] = BINARY_REWARD_VALUE

        # clip reward to -100
        log_r = torch.clamp(log_r, min=BINARY_REWARD_VALUE)
        return log_r


    def _is_tactic_result_an_error(self, tactic_result: str) -> bool:
        for error_string in TACTIC_ERROR_STRINGS:
            if error_string in tactic_result:
                return True
        return False


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
