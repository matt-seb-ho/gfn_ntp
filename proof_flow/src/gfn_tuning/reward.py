from collections import defaultdict
from typing import Optional
from contextlib import contextmanager, nullcontext
import json
import torch
from peft import PeftModel
from transformers import AutoTokenizer
from loguru import logger

from proof_flow.src.constants import (
    DEFAULT_VERIFIER_BATCH_SIZE,
    PROOF_COMPLETE_MESSAGE,
    TACTIC_ERROR_STRINGS
)
from proof_flow.src.utils import (
    batch_iterator_zip,
    causal_conditional_log_prob,
    seq2seq_conditional_log_prob,
)
from proof_flow.src.prompts import RM_TEMPLATES


MIN_REWARD = -30


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
        setup: Optional[str] = None,
        model: Optional[PeftModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        temperature: float = 1.0, 
        batch_size: Optional[int] = None,
        adapter_name: Optional[str] = None,
        seq2seq: bool = False,
        prompts_for_model: Optional[str] = "reprover",
        use_sts_format: bool = False,
        max_input_length: int = 400,
    ):
        self.model = model 
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.batch_size = batch_size or DEFAULT_VERIFIER_BATCH_SIZE
        self.adapter_name = adapter_name
        self.max_input_length = max_input_length

        st_or_sts = "sts" if use_sts_format else "st"
        self.prompt_templates = RM_TEMPLATES[prompts_for_model][st_or_sts]
        
        # select reward computation context and method
        if setup is None:
            self.compute_reward_ctx = nullcontext
            self.compute_log_r = self.compute_binary_log_reward
        else:
            assert model is not None
            assert tokenizer is not None
            self.compute_log_r = self.compute_log_reward
            if setup == "adapter":
                self.compute_reward_ctx = self.adapter_ctx
                assert adapter_name is not None
            elif setup == "base":
                self.compute_reward_ctx = self.base_model_ctx
            elif setup == "independent":
                self.compute_reward_ctx = nullcontext
            else:
                raise ValueError(f"Invalid setup: {setup}")
            
        # select conditional log probability by model architecture
        if seq2seq:
            self.conditional_log_p = seq2seq_conditional_log_prob
        else:
            self.conditional_log_p = causal_conditional_log_prob
            

    def score(
        self,
        states: list[list[str]],
        tactics: list[list[str]],
        batch_size: Optional[int] = None,
        device: Optional[str | torch.device] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            with self.compute_reward_ctx():
                log_reward = self.compute_log_r(
                    states, 
                    tactics, 
                    self.model, 
                    self.tokenizer, 
                    batch_size=(batch_size or self.batch_size),
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
        device: Optional[str | torch.device] = None,
        normalize_tactic_length: bool = True,
        normalize_trajectory_length: bool = True,
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
        prompts = []
        completions = []
        is_partial = torch.zeros(len(states), dtype=torch.bool, device=device)
        trajectory_lengths = torch.tensor(
            [len(t) for t in tactics], 
            device=device, 
            dtype=torch.float32
        )
        for i, (_states, _tactics) in enumerate(zip(states, tactics)):
            # _states: list[str]: represents states for this trajectory
            # _tactics: list[str]: represents tactics for this trajectory
            if _states[-1] == PROOF_COMPLETE_MESSAGE:
                # log_r[i] = 0
                continue
            elif self._is_tactic_result_an_error(_states[-1]):
                log_r[i] = MIN_REWARD
            else:
                # queue prompt-completion logp jobs
                is_partial[i] = True
                for step_idx in range(len(_tactics)):
                    prompt, completion = self._build_reward_inputs(
                        _states[step_idx], 
                        _tactics[step_idx], 
                        _states[step_idx + 1],
                    )
                    trajectory_groups.append(i)
                    prompts.append(prompt)
                    completions.append(completion)
                
        # run queued prompt-completion jobs
        if prompts:
            stepwise_scores = []
            for _prompts, _completions in batch_iterator_zip(
                (prompts, completions), 
                batch_size=batch_size,
            ):
                log_ps, token_lengths = self.conditional_log_p(
                    model, 
                    tokenizer, 
                    _prompts,
                    _completions,
                    max_input_length=self.max_input_length,
                    device=device,
                )
                if normalize_tactic_length:
                    stepwise_scores.append(log_ps / token_lengths)
                else:
                    stepwise_scores.append(log_ps)
                
            stepwise_scores = torch.cat(stepwise_scores)
            log_r = log_r.scatter_add(
                0,                                               # dim
                torch.tensor(trajectory_groups, device=device),  # indices
                stepwise_scores                                  # values
            )

        # normalize partial trajectory scores by trajectory length
        if normalize_trajectory_length:
            scale_factor = torch.where(
                is_partial,
                trajectory_lengths,
                1.0
            )
            log_r /= scale_factor
        
        # logging
        for i, _tactics in enumerate(tactics):
            if not is_partial[i]:
                continue
            t_r = json.dumps({
                "tactics": _tactics,
                "reward": log_r[i].item(),
            })
            logger.info(f"partial reward: {t_r}")

        # clip reward
        log_r = torch.clamp(log_r, min=MIN_REWARD)
        return log_r
    

    def compute_binary_log_reward(
        self,
        states: list[list[str]],
        tactics: list[list[str]],
        model: PeftModel,
        tokenizer: AutoTokenizer,
        batch_size: int = 8,
        device: Optional[str | torch.device] = None,
        normalize_tactic_length: bool = True,
        normalize_trajectory_length: bool = True,
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
                log_r[i] = MIN_REWARD
        # clip reward
        log_r = torch.clamp(log_r, min=MIN_REWARD)
        return log_r


    @contextmanager
    def adapter_ctx(self):
        # policy and reward are adapters over the same model
        was_training = self.model.training
        previously_active_adapter = self.model.active_adapters[0]
        # before: set eval mode, swap to verifier adapter
        self.model.set_adapter(self.adapter_name)
        self.model.eval()
        yield
        # after: swap back to previous adapter, restore training mode
        self.model.set_adapter(previously_active_adapter)
        if was_training:
            self.model.train()
    

    @contextmanager
    def base_model_ctx(self):
        # policy is an adapter over the base model
        # reward is the base model
        was_training = self.model.training
        # before: set eval mode, disable adapters
        self.model.eval()
        with self.model.disable_adapter():
            yield
        # after: restore training mode
        if was_training:
            self.model.train()


    def _is_tactic_result_an_error(self, tactic_result: str) -> bool:
        for error_string in TACTIC_ERROR_STRINGS:
            if error_string in tactic_result:
                return True
        return False


    def _build_reward_inputs(
        self,
        state: str,
        tactic: str,
        next_state: Optional[str] = None,
    ) -> tuple[str, str]:
        return (
            self.prompt_templates["prompt"].format(
                state=state, tactic=tactic, next_state=next_state
            ),
            self.prompt_templates["completion"].format(
                state=state, tactic=tactic, next_state=next_state
            ),
        )
