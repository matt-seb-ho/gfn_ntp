import json
import gc
import random
from contextlib import contextmanager
from typing import Optional
from icecream import ic

import torch
from peft import PeftModel
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

from proof_flow.src.gfn_tuning.proof_tree import (
    ProofTreeNode, extract_trajectories
)
from proof_flow.src.gfn_tuning.replay_buffer import ReplayBuffer
from proof_flow.src.gfn_tuning.reward import NTPReward
from proof_flow.src.prompts import (
    INSTRUCTION_PROMPT_TEMPLATE,
)
from proof_flow.src.utils import (
    prepare_environment_for_lean_dojo,
    batch_iterator_zip,
)


prepare_environment_for_lean_dojo()
from lean_dojo import Dojo, TacticState, Theorem  # isort: skip


class NeuralTheoremProvingTask(LightningModule):
    def __init__(
        self,
        model: PeftModel,
        tokenizer: AutoTokenizer,
        reward: NTPReward,
        reward_buffer: ReplayBuffer,
        n_samples: int | list[int],
        lr: float,
        pf_temp_high: float,
        pf_temp_low: float,
        pf_temp_prob: float,
        use_buffer_prob: float,
        reward_temp_start: float,
        reward_temp_end: float,
        reward_temp_horizon: int,
        use_4bit: bool = False,
        max_tactics: int = 3,
        min_tactic_tokens: int = 2,
        max_tactic_tokens: int = 30,
        use_replay_tree: bool = False,
        model_inference_batch_size: int = 4,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "model", 
            "tokenizer", 
            "reward", 
            "reward_buffer"
        ])

        self.model = model
        self.tokenizer = tokenizer
        self.reward = reward
        self.reward_buffer = reward_buffer
        self.model_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_z = torch.nn.Parameter(
            torch.tensor(0.0, requires_grad=True, device=self.model_device)
        )

        self.get_lr_at_step = lambda step: min(step / 20 * lr, lr)
        self.get_reward_temp_at_step = lambda step: reward_temp_start + (
            reward_temp_end - reward_temp_start
        ) * min(1, step / reward_temp_horizon)

        # end step generation at newline char
        self.end_of_step_token_id = tokenizer.encode("assumption\n", add_special_tokens=False)[-1]
    

    def forward(
        self, 
        theorem: Theorem, 
        n_samples: Optional[int | list[int]] = None, 
        pf_temperature: float =  1.0, 
        max_depth: int = 3, # depth starts at 0
        replay_tactics: Optional[ProofTreeNode] = None,
        generate_func: Optional[callable] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list]]:
        """
        Samples a proof tree for a given theorem
        Returns
            - trajectories_logpf: (n_samples, max_depth) tensor of log probabilities of each tactic
            - log_r: (n_samples,) tensor of log rewards for complete trajectories
            - extracted_trajectories: list of trajectories for the replay buffer
        """
        trajectories_logpf: list[torch.Tensor] = []
        log_reward: list[float] = []

        # n_samples[d] is the number of tactics to sample at depth d
        # if provided as int, we use the same number of samples at each depth
        if not isinstance(n_samples, list):
            n_samples = [n_samples or self.hparams.n_samples] * max_depth
        with lean_context(theorem, replay_tactics) as (dojo, root):
            trajectory_logpf: list[torch.Tensor] = []
            stack = [(root, False)]
            while stack:
                node, visited = stack.pop()
                # node should not be terminal
                # assert node.depth < max_depth and isinstance(node, TacticState)
                if visited:
                    # backing out: clean up this node's element from the trajectory
                    # node.tactic_logpf is None for the root node, so trajectory_logpf is empty
                    if node.tactic_logpf is not None:
                        trajectory_logpf.pop()
                    continue
                # backtracking purposes
                stack.append((node, True))
                if node.tactic_logpf is not None:
                    trajectory_logpf.append(node.tactic_logpf)
                self.expand_node(
                    node, 
                    n_samples=n_samples[node.depth], 
                    pf_temperature=pf_temperature,
                    max_depth=max_depth,
                    lean_env=dojo,
                    replay=(replay_tactics is not None),
                    generate_func=generate_func,
                    device=self.model_device,
                )
                new_stack_items = []
                for child in node.children:
                    if node.depth + 1 < max_depth and isinstance(child.state, TacticState):
                        new_stack_items.append((child, False))
                    else:
                        # terminal
                        trajectories_logpf.append(torch.cat(trajectory_logpf + [child.tactic_logpf]))
                        log_reward.append(child.log_r)
                # add new stack items in reverse order for depth-first traversal
                # - why not iterate reversed(node.children)?
                #   we want to handle the terminal outputs in the correct order
                stack.extend(reversed(new_stack_items))
                
        
        # trajectories can have different lengths (may be jagged) and need to be padded
        trajectories_logpf = pad_sequence(
            trajectories_logpf, 
            batch_first=True, 
            padding_value=0.0
        )
        if replay_tactics:
            log_reward = torch.tensor(log_reward)
            trajectories = None
        else:
            # compute and assign log_r to leaf nodes
            trajectories = extract_trajectories(root, theorem.uid)
            states = [t["states"] for t in trajectories]
            tactics = [t["tactics"] for t in trajectories]
            log_reward = self.reward.score(states, tactics, device=self.model_device)

            # trajectories were extracted from the tree before log_r was computed
            # - ensure we have the log_r for each trajectory for the replay buffer
            for trajectory, log_r in zip(trajectories, log_reward):
                trajectory["log_r"] = log_r.item()

        return trajectories_logpf, log_reward, trajectories


    def tb_loss(self, log_pf: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
        """
        Computes the batch loss using the Trajectory Balance objective

        Args:
            log_pf: log probabilities of each tactic/action. shape=(batch, max_depth)
            log_r: log reward. shape=(batch_size,)

        Returns:
            batch_loss: scalar tensor
        """
        loss = (log_pf.sum(dim=-1) + self.log_z - log_r) ** 2
        batch_loss = loss.sum()
        return batch_loss

    
    def log_z_variance_loss(log_pf: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
        # log_pf has shape (batch_size, max_tactic_depth)
        # log_r has shape (batch_size,)
        # https://arxiv.org/pdf/2302.05446
        trajectory_log_pf = log_pf.sum(dim=-1)
        batch_zeta = log_r - trajectory_log_pf
        expectation = batch_zeta.mean()
        return ((batch_zeta - expectation) ** 2).sum()


    def training_step(
        self, 
        theorem: Theorem, 
        batch_idx: int,              # required by PyTorch Lightning(?)
        force_replay: bool = False,  # for testing purposes
):
        theorem_id = theorem.uid

        # replay trajectories
        # _sample... helper function handles the logic of whether to use the buffer or not
        # uses buffer if:
        #   (1) random() < hparams.use_buffer_prob 
        #   (2) the theorem has trajectories in the buffer
        replay_ts = self._sample_replay_trajectories(theorem_id, force_replay=force_replay)
        
        if replay_ts is not None:
            # using sample from replay buffer
            if self.hparams.use_replay_tree:
                t_logpf, log_r, _ = self.forward(theorem, replay_tactics=replay_ts)
            else:
                t_logpf, log_r, = self.replay_trajectories(
                    replay_ts,
                    model_inf_batch_size=self.hparams.model_inference_batch_size,
                )
            log_r *= 1 / self.reward.temperature  # redo the effect of reward tempering
        else:
            # Using the forward policy
            if random.random() < self.hparams.pf_temp_prob:  # With tempering
                pf_temp = (
                    random.random()
                    * (self.hparams.pf_temp_high - self.hparams.pf_temp_low)
                    + self.hparams.pf_temp_low
                )
            else:
                # Without tempering
                pf_temp = 1.0

            t_logpf, log_r, extracted_ts = self.forward(theorem, pf_temperature=pf_temp)
            self.reward_buffer.add_batch(theorem_id, extracted_ts)

        # get gfn loss
        # - sub tb requires estimating flow (possible impl: scalar head over RM)
        # - for the proof of concept, we'll just use vanilla TB
        loss = self.tb_loss(log_pf=t_logpf, log_r=log_r)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "train/logR",
            # last_log_r.mean(),
            log_r.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss


    def validation_step(self, theorem: Theorem, batch_idx: int):
        # sample a proof and get the reward
        log_pf, log_r, _ = self.forward(theorem)

        # get the GFN loss
        loss = self.tb_loss(log_pf=log_pf, log_r=log_r)

        # Log metrics
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val/logR",
            log_r.mean(),
            sync_dist=True,
        )


    def on_train_batch_start(self, theorem, batch_idx):
        # Update scheduled quantities
        reward_temp = self.get_reward_temp_at_step(self.global_step)
        lr = self.get_lr_at_step(self.global_step)
        self.reward.temperature = reward_temp
        for pg in self.optimizers().param_groups:
            pg["lr"] = lr


    def on_train_epoch_start(self):
        # Log scheduled quantities
        self.log("scheduled/R_temperature", self.reward.temperature, sync_dist=True)
        self.log("scheduled/lr", self.get_lr_at_step(self.global_step), sync_dist=True)


    def configure_optimizers(self):
        if self.hparams.use_4bit:
            import bitsandbytes as bnb  # fmt: skip
            return bnb.optim.PagedAdamW8bit(self.model.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
    

    def expand_node(
        self,
        node: ProofTreeNode,
        n_samples: Optional[int] = None,
        pf_temperature: float = 1.0,
        max_depth: int = 3,
        lean_env: Optional[Dojo] = None,
        replay: bool = False,
        generate_func: Optional[callable] = None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        """
        Expands a node in the proof tree by generating tactics.
        If replaying a trajectory, the method recomputes log_pf with the updated model.
        """
        assert (lean_env is not None) or replay, "lean_env must be provided if not replaying tactics"
        if (node.state and not isinstance(node.state, TacticState)) or node.depth >= max_depth:
            return
        if replay:
            # replay tree construction should have already prepared the input_ids
            input_ids = node.children_tactic_tokens
        else:
            # generate n new tactics
            prompt_text = self.format_prompt(node.state.pp if node.state else node.state_str) 
            input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            n_samples = n_samples or self.hparams.n_samples
            input_ids = input_ids.expand(n_samples, -1)
            prompt_length = input_ids.shape[1]
            if device:
                input_ids = input_ids.to(device)

        # sample tactics (if not provided by action_seq) and compute terms needed for loss
        if generate_func and not replay:
            # for baseline generation/decoding methods
            tokens, log_pf = generate_func(input_ids)
        else:
            tokens, log_pf = generate_step(
                self.model,
                input_ids,
                termination_token_id=self.end_of_step_token_id,
                temperature=pf_temperature,
                replay=replay,
                prompt_length=prompt_length,
                max_new_tokens=self.hparams.max_tactic_tokens,
            )

        # log_pf outputted by generate_step is at token-level granularity
        # NOTE: this assumes that padded tokens have a log_pf of 0
        log_pf_tactic = log_pf.sum(dim=1)
        
        if replay:
            for child, tactic_logpf in zip(node.children, log_pf_tactic):
                child.tactic_logpf = tactic_logpf
        else:
            # SKIPPED: replay uses reconstructed trees ~~save the generated tactics for replay~~
            # node.children_tactic_tokens = tokens
            # pass the generated tactics through the lean environment
            generated_tactics = self.tokenizer.batch_decode(
                tokens[:, input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            # tactics_token_length = (tokens[:, input_ids.shape[1]:] == self.end_of_sentence_token_id).count_nonzero(dim=-1)
            # looking for first occurence of end_of_step_token_id after the prompt
            # - among multiple max values, argmax returns the first occurrence
            # - this excludes the end of step token (eos token omitted in replay buffer!!)
            is_end_of_step = (tokens[:, prompt_length:]).eq(self.end_of_step_token_id)
            pre_pad_length = is_end_of_step.float().argmax(dim=-1) + prompt_length

            # create new children nodes by running the generated tactics through the environment
            for i, tactic in enumerate(generated_tactics):
                next_state = lean_env.run_tac(node.state, tactic.rstrip())
                child_node = ProofTreeNode(
                    state=next_state,
                    tactic=generated_tactics[i].strip(),
                    depth=node.depth + 1,
                    prompt_length=input_ids.shape[1],
                    tactic_logpf=log_pf_tactic[i:i+1],
                    parent_tactic_tokens=tokens[i, :pre_pad_length[i]].cpu(),
                    parent=node,
                )
                if node.children is None:
                    node.children = []
                node.children.append(child_node)
    

    def replay_trajectories(
        self,
        trajectories: list[dict],
        model_inf_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Replay a batch of trajectories, computing new log_pf for tactics.
        Effectively the replacement for `forward` when replaying trajectories.

        Returns
            - t_log_pf: (n_samples, max_depth) tensor of log probabilities of each tactic (0-padded).
            - log_r: (n_samples,) tensor of log rewards for complete trajectories
        """
        # with the current tb loss formulation, we can actually get away with
        # preemptively summing the tactic log_pfs to get the trajectory's log_pf
        # but for now, we'll keep the trajectory log_pf separate
        max_depth = max(len(t["state_tactic_tokens"]) for t in trajectories)
        step_logpfs = torch.zeros(
            # (len(trajectories), self.hparams.max_tactics), 
            (len(trajectories), max_depth),
            device=self.model_device
        )
        # eager sum version
        # t_logpfs = torch.zeros(len(trajectories), device=self.model_device)

        # queue up jobs
        input_ids = []
        prompt_lengths = []
        batch_idxs = []
        step_idxs = []
        for b_idx, trajectory in enumerate(trajectories):
            for s_idx, (tokens, prompt_length) in enumerate(zip(
                trajectory["state_tactic_tokens"],
                trajectory["prompt_lengths"],
            )):
                input_ids.append(tokens)
                prompt_lengths.append(prompt_length)
                batch_idxs.append(b_idx)
                step_idxs.append(s_idx)
        
        # compute tactic log_pfs in batches
        for tokens, prefix_lengths, b_idxs, s_idxs in batch_iterator_zip(
            (input_ids, prompt_lengths, batch_idxs, step_idxs), 
            batch_size=model_inf_batch_size
        ):
            batch_input_ids = pad_sequence(
                tokens,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            step_logpfs[b_idxs, s_idxs] = self._compute_replay_log_pfs(
                batch_input_ids,
                torch.tensor(prefix_lengths),
                self.end_of_step_token_id,
                self.tokenizer.pad_token_id,
                device=self.model_device,
            )
            # eager sum version
            # batch_res = self._get_completion_log_pfs(
            #     batch_input_ids,
            #     torch.tensor(prefix_lengths),
            #     self.end_of_step_token_id,
            #     self.tokenizer.pad_token_id,
            #     device=self.model_device,
            # )
            # idx = torch.tensor(b_idxs, device=self.model_device)
            # t_logpfs = t_logpfs.scatter_add(0, idx, batch_res)
        
        # collect log_r from replay buffer
        log_r = torch.tensor(
            [t["log_r"] for t in trajectories], 
            device=self.model_device
        )
        return step_logpfs, log_r
        # eager sum version
        # return t_logpfs, log_r
    

    def _compute_replay_log_pfs(
        self,
        input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        termination_token_id: int,
        pad_token_id: int,
        device: Optional[str | torch.device] = None,
    ) -> torch.Tensor:
        # essentially alternate input version of batch_completion_probabilities
        # critical assumptions
        # - model is already in policy mode (policy adapters active)
        attention_mask = (input_ids != pad_token_id).long()
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            prompt_lengths = prompt_lengths.to(device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        log_prob_distributions = torch.log_softmax(outputs.logits, dim=-1)
        # distribution after last token can be ignored
        log_prob_distributions = log_prob_distributions[:, :-1, :]
        # can't get log probabilities for first token
        relevant_tokens = input_ids[:, 1:]
        seq_log_probs = torch.gather(
            log_prob_distributions, 
            # dim explanation: shape is (batch, tokens, vocab) -> gather on vocab
            2, 
            relevant_tokens[:, :, None]
        ).squeeze(-1)

        # start: -1 because the input_ids were shifted forward by 1
        start = (prompt_lengths - 1).unsqueeze(1)

        # stop = (relevant_tokens == termination_token_id).float().argmax(dim=-1).unsqueeze(1)
        # this old implementation doesn't work for two reasons
        # 1. term token can appear in the prompt and prompt length is variable
        #    making the search a little difficult
        # 2. preprocessing state tactic tokens before adding to the
        #    replay buffer INCLUDES REMOVING END OF STEP TOKEN IN ADDITION TO PADDING
        # fixed implementation depends only on pad tokens
        pad_occurrences = (relevant_tokens == pad_token_id)
        first_pad_idx = torch.where(
            pad_occurrences.any(dim=1), 
            pad_occurrences.float().argmax(dim=1), 
            relevant_tokens.shape[1]
        )
        stop = (first_pad_idx).unsqueeze(1)

        # create an index tensor for the sequence dimension
        # [0, 1, 2, ..., seq_len - 1] for every batch element
        # shape (batch_size, seq_len)
        idx = (
            torch.arange(seq_log_probs.shape[1], device=seq_log_probs.device)
            .unsqueeze(0)
            .expand(seq_log_probs.shape[0], -1)
        )
        # create the mask based on the condition start <= idx < stop
        # mask is 1 where we want to keep the log probabilities
        mask = (idx >= start) & (idx < stop)
        return (seq_log_probs * mask).sum(dim=1)
    

    def _sample_replay_trajectories(
        self, 
        theorem_id: str,
        force_replay: bool = False,
    ) -> Optional[ProofTreeNode | list[dict]]:
        # if force replay is True, this random check is skipped
        if (not force_replay and random.random() >= self.hparams.use_buffer_prob):
            return None
        if self.hparams.use_replay_tree:
            return self.reward_buffer.sample_tree(theorem_id, self.hparams.n_samples)
        else:
            return self.reward_buffer.sample(theorem_id, self.hparams.n_samples)


    @staticmethod
    def format_prompt(state: str):
        # TODO: verify Dojo's pp output is the "goals:..."
        # prepending "-- " to every line to match the evaluation setup in Llemma-7B paper
        # - see figure 4
        # appending a newline to the end of the prompt if it doesn't exist
        # lines = state.split("\n")
        # lines[-1] = lines[-1].rstrip() + "\n"
        # # TODO: check if we want line.lstrip() 
        # commented_lines = ["-- INPUT:"] + ["-- " + line for line in lines]
        # return "\n".join(commented_lines)
        return INSTRUCTION_PROMPT_TEMPLATE.format(state=state)


def generate_step(
    model: PeftModel,
    input_ids: torch.Tensor,
    termination_token_id: int,
    temperature: float = 1.0,
    top_k: int = 999999,
    top_p: float = 1.0,
    replay: bool = False,
    prompt_length: Optional[int] = None,
    max_new_tokens: int = 30,
    **generation_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    forward pass for a single state
    returns:
    1. token id tensor of (state, tactic)
    2. log probability tensor for the tactics
    
    if replay is True, then the passed input_ids should include the previously generated tactics
    """
    if replay:
        # assert prompt_length is not None
        outputs = model(
            input_ids=input_ids,
            eos_token_id=termination_token_id,
            return_dict=True,
        )
        # the standard usage of compute_transition_scores is to pass GenerateOutput.scores
        # which is a tuple (length=max_seq_len) of tensors of shape (batch_size, vocab_size).
        # - we need to rearrange the model.__call__ logits to match this shape
        input_scores = tuple(outputs.logits.transpose(0, 1)[prompt_length-1:-1])
        transition_scores = model.compute_transition_scores(
            input_ids,
            input_scores,
            normalize_logits=True,
        )
        # post processing
        # - mask out pad tokens 
        pad_mask = (input_ids[:, prompt_length:] == model.config.pad_token_id)
        transition_scores[pad_mask] = 0
        return input_ids, transition_scores
        
    # model.generate notes
    # - do_sample=True turns off greedy decoding
    # - begin_suppress_tokens: prevents the first token from being the termination token
    #   - this prevents empty tactics from being generated (reward current cannot handle empty tactics)
    # - passing in a different eos_token_id acts as a stopping criteria 
    #   - (attention computations are not affected)
    # - choosing to output logits instead of scores to get behaviour more similar
    #   to the replay computation (scores are different because of suppress)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=termination_token_id,
        forced_eos_token_id=termination_token_id,
        begin_suppress_tokens=[termination_token_id],
        return_dict_in_generate=True,
        output_logits=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        **generation_kwargs
    )
    scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.logits,
        normalize_logits=True,
    )
    # the original generate_and_return_termination_logprob returned (state, log_pf, ...)
    # where state includes the prompt and the generated tokens, 
    # while log_pf included only the generated tokens' log probabilities.
    prompt_length = prompt_length or input_ids.shape[1]
    # fix padding:
    # - pad out scores from term token onwards (first term token after prompt)
    pad_mask = (
        outputs.sequences[:, prompt_length:] == termination_token_id
    ).cumsum(dim=1) > 0
    scores[pad_mask] = 0
    return outputs.sequences, scores


@contextmanager
def lean_context(theorem: Theorem, replay_tactics: Optional[ProofTreeNode] = None):
    if replay_tactics:
        yield None, replay_tactics
    else:
        with Dojo(theorem) as (dojo, initial_state):
            new_root = ProofTreeNode(state=initial_state, children=[])
            yield dojo, new_root
