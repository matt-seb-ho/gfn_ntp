import gc
import random
from contextlib import contextmanager
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

from proof_flow.src.gfn_tuning.proof_tree import ProofTreeNode, extract_trajectories
from proof_flow.src.gfn_tuning.replay_buffer import ReplayBuffer
from proof_flow.src.gfn_tuning.reward import NTPReward, compute_log_reward
from proof_flow.src.gfn_tuning.verifier import batch_iterator
from proof_flow.src.utils import (
    base_to_lora, 
    lora_to_base,
    prepare_environment_for_lean_dojo,
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
        illegal_token_mask: np.ndarray,
        train_probes: Optional[list[Theorem]] = None,
        val_probes: Optional[list[Theorem]] = None,
        use_4bit: bool = False,
        max_tactics: int = 3,
        min_tactic_tokens: int = 2,
        max_tactic_tokens: int = 30,
        use_replay_tree: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer"])

        self.model = model
        self.tokenizer = tokenizer
        self.log_z = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
        self.reward = reward
        self.reward_buffer = reward_buffer

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
        leaf_nodes: list[ProofTreeNode] = []
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
                )
                for child in reversed(node.children):
                    if node.depth + 1 < max_depth and isinstance(child.state, TacticState):
                        stack.append(child)
                    else:
                        # terminal
                        trajectories_logpf.append(torch.cat(trajectory_logpf + [child.tactic_logpf]))
                        leaf_nodes.append(child)
                        log_reward.append(child.log_r)
        
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
            log_reward = self.reward.score(states, tactics)
            for leaf_node, log_r in zip(leaf_nodes, log_reward):
                leaf_node.log_r = log_r.item()
        return trajectories_logpf, log_reward, trajectories


    def tb_loss(self, log_pf: torch.Tensor, log_r: torch.Tensor):
        """
        Computes the batch loss using the Trajectory Balance objective

        Arguments
            - log_pf: Tensor of shape (batch_size, max_depth)
            - log_r: Tensor of shape (batch_size,)
        """
        loss = (log_pf.sum(dim=-1) + self.log_z - log_r) ** 2
        batch_loss = loss.sum()
        return batch_loss


    def training_step(self, theorem: Theorem, batch_idx):
        theorem_id = theorem.uid

        # replay trajectories
        # _sample... helper function handles the logic of whether to use the buffer or not
        #   (1) hparams.use_buffer_prob 
        #   (2) the theorem has trajectories in the buffer
        replay_ts = self._sample_replay_trajectories(theorem_id)
        
        if replay_ts is not None:
            # using sample from replay buffer
            if self.hparams.use_replay_tree:
                t_logpf, log_r, _ = self.forward(theorem, replay_tactics=replay_ts)
            else:
                t_logpf, log_r, = self.replay_trajectories(
                    replay_ts,
                    model_inf_batch_size=self.hparams.model_inf_batch_size,
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

        # Log probe samples
        if (
            self.hparams.train_probes is not None
            and self.logger is not None
            and self.trainer.current_epoch % 5 == 0
        ):
            samples_table = self.sample_probes(self.hparams.train_probes)
            self.logger.log_table("samples/train_probes", dataframe=samples_table)


    def on_validation_epoch_start(self):
        """
        They perform inference over the validation set (again), albeit with temp=2.0
        Seems this is done to get variance of (logR - logPf(s))
        - I am not certain of how important this is to keep for our task
          I assume since we have a different reward function, this may not be needed.

        Additionally, every 5 epochs, they log the samples from the validation probes
        """
        # Log variance of (logR - logP(s)) using exploration, which should be 0.0
        # log_rs, log_pfss = [], []
        # val_data = self.trainer.datamodule.val_dataloader().dataset
        # for prompt in val_data:
        #     prompt = prompt[0]
        #     generated_text, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
        #         prompt.to(self.device), pf_temperature=2.0
        #     )
        #     log_pfs, log_r, _, _ = get_termination_vals(
        #         generated_text=generated_text,
        #         log_pf=log_pf,
        #         log_pterm=log_pterm,
        #         log_r=log_r,
        #         log_r_unpenalized=log_r_unpenalized,
        #         termination_token_id=self.end_of_sentence_token_id,
        #         prompt_len=len(prompt),
        #     )
        #     log_rs.append(log_r)
        #     log_pfss.append(log_pfs)
        # log_rs, log_pfss = torch.cat(log_rs), torch.cat(log_pfss)
        # self.log("val/Var(logR - logPf(s))", (log_rs - log_pfss).var(), sync_dist=True)

        # log probe samples
        if (
            self.hparams.val_probes is not None
            and self.logger is not None
            and self.trainer.current_epoch % 5 == 0
        ):
            samples_table = self.sample_probes(self.hparams.val_probes)
            self.logger.log_table("samples/val_probes", dataframe=samples_table)


    def on_train_start(self):
        # log baseline metrics
        if self.logger is None:
            return

        val_data = self.trainer.datamodule.val_dataloader().dataset
        baseline_performance = None
        for theorem in val_data:
            samples = self.sample_baselines(theorem, n_samples=self.hparams.n_samples)
            if baseline_performance is None:
                table_index = [
                    "logP(s) (avg)",
                    "logP(s) (max)",
                    "logR (avg)",
                ]
                baseline_performance = pd.DataFrame(
                    data=np.zeros((len(table_index), len(samples))),
                    columns=samples.keys(),
                    index=table_index,
                )
            for baseline in samples:
                baseline_performance.loc["logP(s) (avg)", baseline] += (
                    samples[baseline]["logP(s)"].mean().item() / len(val_data)
                )
                baseline_performance.loc["logP(s) (max)", baseline] += (
                    samples[baseline]["logP(s)"].max().item() / len(val_data)
                )
                baseline_performance.loc["logR (avg)", baseline] += (
                    samples[baseline]["logR"].mean().item() / len(val_data)
                )
        baseline_performance = baseline_performance.reset_index(names="metric")
        self.logger.log_table("val/baseline performance", dataframe=baseline_performance)

        # log baseline probes
        if self.hparams.val_probes is not None:
            samples_table = self.sample_probes_baselines(self.hparams.val_probes)
            self.logger.log_table(
                "samples/val_probes (baselines)", dataframe=samples_table
            )

    def sample_probes(self, probes, n_samples=4):
        assert isinstance(probes, list) and probes[0].ndim == 1
        samples = []
        for probe in probes:
            with torch.no_grad():
                log_pf, log_r, extracted_ts = self.forward(probe, n_samples=n_samples)
            log_pfs = log_pf.sum(dim=-1)
            for i, trajectory in enumerate(extracted_ts):
                samples.append(
                    {
                        "theorem": probe.full_name,
                        "theorem id": probe.uid,
                        "sampled proof": trajectory["proof"],
                        "final_state": trajectory["states"][-1],
                        "logP(s)": log_pfs[i].item(),
                        "logR": log_r[i].item(),
                        # "logP(s) unpenalized": log_ps_unpenalized[i].item(),
                    }
                )
        samples = pd.DataFrame(samples)
        samples = samples.sort_values(by=["theorem", "logR"], ascending=False)
        return samples

    def sample_probes_baselines(self, probes, n_samples=4):
        assert isinstance(probes, list) and isinstance(probes[0], Theorem)
        samples = []
        for probe in probes:
            probe_samples = self.sample_baselines(probe, n_samples=n_samples)
            for baseline in probe_samples:
                for sample in probe_samples[baseline]:
                    sample["baseline"] = baseline
                    samples.append(sample)

        samples = pd.DataFrame(samples)
        samples = samples.sort_values(by=["theorem"], ascending=False)
        return samples

    def sample_baselines(self, theorem, n_samples=4):
        # https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationMixin.generate
        # https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationConfig
        assert prompt.ndim == 1
        prompt = prompt.unsqueeze(0)

        def generate_func(prompt, **kwargs):
            with torch.no_grad():
                lora_to_base(self.model)
                suppressed_tokens = torch.from_numpy(self.hparams.illegal_token_mask).nonzero().squeeze(-1)
                outputs = self.model.generate(
                    prompt,
                    min_new_tokens=self.hparams.min_sentence_len,
                    max_new_tokens=self.hparams.max_sentence_len + 1,
                    eos_token_id=self.end_of_sentence_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    forced_eos_token_id=self.end_of_sentence_token_id,
                    suppress_tokens=suppressed_tokens,
                    **kwargs,
                )
                base_to_lora(self.model)
                tokens = outputs.sequences
                tokens = torch.where(
                    tokens == self.tokenizer.eos_token_id,
                    self.end_of_step_token_id,
                    tokens,
                )
                probs = torch.log_softmax(outputs.logits, dim=-1).detach()
                probs = probs[:, :-1, :]
                input_ids = outputs.input_ids[:, 1:]
                gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
                end_mask = (input_ids == self.end_of_step_token_id).byte()
                gen_lengths = end_mask.argmax(dim=-1).unsqueeze(-1)
                log_pf = torch.gather(gen_probs.cumsum(dim=-1), -1, gen_lengths).squeeze(-1)

                return outputs, log_pf

        sample_kwargs = {
            "beam": {"do_sample": False, "num_beams": n_samples * 5, "length_penalty": 0.0},
            "beam [fair]": {"do_sample": False, "num_beams": n_samples, "length_penalty": 0.0},
            "diverse beam": {
                "num_beams": n_samples * 5, 
                "num_beam_groups": n_samples, 
                "num_return_sequences": n_samples, 
                "diversity_penalty": 1.0, 
                "length_penalty": 0.0
            },
            "diverse beam [fair]": {
                "num_beams": n_samples, 
                "num_beam_groups": n_samples, 
                "num_return_sequences": n_samples, 
                "diversity_penalty": 1.0, 
                "length_penalty": 0.0
            },
            "nucleus": {
                "do_sample": True, 
                "num_return_sequences": n_samples, 
                "top_k": 0, 
                "top_p": 0.95
            },
            "LM": {"do_sample": True, "num_return_sequences": n_samples, "top_k": 0},
            "LM tempered": {
                "do_sample": True, 
                "num_return_sequences": n_samples, 
                "top_k": 0, 
                "temperature": self.hparams.reward_temp_end
            },
            "greedy": {"do_sample": False},
        }

        samples = {}
        for sample_method, kwargs in sample_kwargs.items():
            generation_func = partial(generate_func, **kwargs)
            trajectories_logpf, log_r, extracted_ts = self.forward(
                theorem, 
                n_samples=n_samples,
                generate_func=generation_func,
            )
            log_pfs = trajectories_logpf.sum(dim=-1)
            samples[sample_method] = [
                {
                    "theorem": theorem.full_name,
                    "theorem id": theorem.uid,
                    "sampled proof": trajectory["proof"],
                    "final_state": trajectory["states"][-1],
                    "logP(s)": log_pfs[i].item(),
                    "logR": log_r[i].item(),
                }
                for i, trajectory in enumerate(extracted_ts)
            ]

        return samples

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
            input_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            n_samples = n_samples or self.hparams.n_samples
            input_ids = input_ids.expand(n_samples, -1)

        # sample tactics (if not provided by action_seq) and compute terms needed for loss
        if generate_func and not replay:
            # for baseline generation/decoding methods
            tokens, log_pf = generate_func(input_ids)
        else:
            try:
                tokens, log_pf = generate_step(
                    self.model,
                    input_ids,
                    termination_token_id=self.end_of_step_token_id,
                    temperature=pf_temperature,
                    replay=replay,
                    prompt_length=node.prompt_length,
                    max_new_tokens=self.hparams.max_tactic_tokens,
                )
            except RuntimeError as e:
                if "out of memory" not in str(e):
                    raise e
                # clean up memory and return (cuts trajectory short)
                torch.cuda.empty_cache()
                gc.collect()
                return

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
            # among multiple max values, argmax returns the first occurrence
            # this excludes the end of step token
            pre_pad_length = (tokens == self.end_of_step_token_id).argmax(dim=-1)

            # create new children nodes by running the generated tactics through the environment
            for i, tactic in enumerate(generated_tactics):
                next_state = lean_env.run_tac(node.state, tactic.rstrip())
                child_node = ProofTreeNode(
                    state=next_state,
                    tactic=generated_tactics[i].strip(),
                    depth=node.depth + 1,
                    prompt_length=input_ids.shape[1],
                    tactic_logpf=log_pf_tactic[i],
                    parent_tactic_tokens=tokens[i, :pre_pad_length[i]],
                    parent=node,
                )
                if node.children is None:
                    node.children = []
                node.children.append(child_node)
    

    def replay_trajectories(
        self,
        replay_trajectories: list[dict],
        model_inf_batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Replay a batch of trajectories, computing new log_pf for tactics.
        Effectively the replacement for `forward` when replaying trajectories.

        Returns
            - t_log_pf: (n_samples, max_depth) tensor of log probabilities of each tactic (0-padded).
            - log_r: (n_samples,) tensor of log rewards for complete trajectories
        """

        # queue up jobs
        jobs = []
        idxs = []
        for t_idx, trajectory in enumerate(replay_trajectories):
            for tokens, prompt_length in zip(
                trajectory["state_tactic_tokens"],
                trajectory["prompt_lengths"],
            ):
                jobs.append((tokens, prompt_length))
                idxs.append(t_idx)
        stepwise_log_pfs = []
        
        # compute tactic log_pfs in batches
        for batch in batch_iterator(jobs, batch_size=model_inf_batch_size):
            batch_input_ids = pad_sequence(
                [tokens for tokens, _ in batch],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            stepwise_log_pfs.append(
                self._get_completion_log_pfs(
                    batch_input_ids,
                    torch.tensor([pl for _, pl in batch]),
                    self.end_of_step_token_id,
                    self.tokenizer.pad_token_id,
                )
            )
        # combine by idx (trajectory idx) using scatter_add
        idxs = torch.tensor(idxs)
        stepwise_log_pfs = torch.cat(stepwise_log_pfs)
        log_pf = torch.zeros(len(replay_trajectories), dtype=stepwise_log_pfs.dtype)
        log_pf = log_pf.scatter_add(0, idxs, stepwise_log_pfs)
        
        # collect log_r from replay buffer
        log_r = torch.tensor([t["log_r"] for t in replay_trajectories])
        return log_pf, log_r
    

    def _get_completion_log_pfs(
        self,
        input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        termination_token_id: int,
        pad_token_id: int,
        device: Optional[torch.device] = None,
        split_and_retry: bool = True,
    ) -> torch.Tensor:
        # essentially alternate input version of batch_completion_probabilities
        # critical assumptions
        # - model is already in policy mode (policy adapters active)
        attention_mask = (input_ids != pad_token_id).long()
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        except RuntimeError as e:
            if "out of memory" not in str(e) or not split_and_retry:
                raise e
            # split the job in half and retry!
            sub_results = [
                self._get_completion_log_pfs(
                    half,
                    prompt_lengths,
                    termination_token_id,
                    pad_token_id,
                    device=device,
                    split_and_retry=False,
                )
                for half in input_ids.split(input_ids.shape[0] // 2)
            ]
            return torch.cat(sub_results)
            
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

        # -1 is because the input_ids were shifted forward by 1
        start = (prompt_lengths - 1).unsqueeze(1)
        # exclude the end token
        stop = (relevant_tokens == termination_token_id).argmax(dim=-1).unsqueeze(1)
        # create an index tensor for the sequence dimension
        # [0, 1, 2, ..., seq_len - 1] for every batch element
        # shape (batch_size, seq_len)
        idx = (
            torch.arange(seq_log_probs.shape[1])
            .unsqueeze(0)
            .expand(seq_log_probs.shape[0], -1)
        )
        # create the mask based on the condition start <= idx < stop
        # mask is 1 where we want to keep the log probabilities
        mask = (idx >= start) & (idx < stop)
        return (seq_log_probs * mask).sum(dim=1)
    

    def _sample_replay_trajectories(
        self, 
        theorem_id: str
    ) -> Optional[ProofTreeNode | list[dict]]:
        if (random.random() >= self.hparams.use_buffer_prob):
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
        lines = state.split("\n")
        lines[-1] = lines[-1].rstrip() + "\n"
        # TODO: check if we want line.lstrip() 
        commented_lines = ["-- INPUT:"] + ["-- " + line for line in lines]
        return "\n".join(commented_lines)


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
        try:
            outputs = model(
                input_ids=input_ids,
                eos_token_id=termination_token_id,
                return_dict=True,
            )
        except RuntimeError as e:
            if "out of memory" not in str(e):
                raise e
            # on replay, we know that we can handle these inputs
            # so we can split the job in half and retry
            halves = input_ids.split(input_ids.shape[0] // 2)
            log_pf_sub_results = [
                generate_step(
                    model,
                    half,
                    termination_token_id=termination_token_id,
                    temperature=temperature,
                    replay=replay,
                    prompt_length=prompt_length,
                    max_new_tokens=max_new_tokens,
                )[1] 
                for half in halves
            ]
            log_pf = torch.cat(log_pf_sub_results)
            return input_ids, log_pf

        # the standard usage of compute_transition_scores is to pass GenerateOutput.scores
        # which is a tuple (length=max_seq_len) of tensors of shape (batch_size, vocab_size).
        # - we need to rearrange the model.__call__ logits to match this shape
        scores = tuple(outputs.logits.transpose(0, 1)[prompt_length-1:-1])
        scores = model.compute_transition_scores(
            input_ids,
            scores,
            normalize_logits=True,
        )
        # post processing
        # - mask out pad tokens 
        pad_mask = (input_ids[:, prompt_length:] == model.config.pad_token_id)
        scores[pad_mask] = 0
        return input_ids, scores
        
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=termination_token_id,
        forced_eos_token_id=termination_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        **generation_kwargs
    )
    scores = model.compute_transition_scores(
        outputs.sequences,
        outputs.scores,
        normalize_logits=True,
    )
    # the original generate_and_return_termination_logprob returned (state, log_pf, ...)
    # where state includes the prompt and the generated tokens, 
    # while log_pf included only the generated tokens' log probabilities.
    pad_mask = outputs.sequences == model.config.pad_token_id
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
