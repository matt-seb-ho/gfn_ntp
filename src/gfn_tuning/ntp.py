import random
from functools import partial
from typing import Optional
import pandas as pd
import numpy as np
import torch
from contextlib import contextmanager
from pytorch_lightning import LightningModule
from utils import (
    modified_subtb_loss,
    get_termination_vals,
)
from utils import lora_to_base, base_to_lora
from lean_dojo import (
    Dojo,
    TacticResult,
    TacticState, 
    Theorem,
    ProofFinished,
)
from proof_tree import ProofTreeNode, extract_trajectories
from replay_buffer import ReplayBuffer
from reward import compute_log_reward


class NeuralTheoremProvingTask(LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        reward,
        reward_buffer: ReplayBuffer,
        n_samples,
        lr,
        subtb_lambda,
        pf_temp_high,
        pf_temp_low,
        pf_temp_prob,
        use_buffer_prob,
        min_sentence_len,
        max_sentence_len,
        reward_temp_start,
        reward_temp_end,
        reward_temp_horizon,
        illegal_token_mask,
        train_probes=None,
        val_probes=None,
        use_4bit=False,
        max_steps=3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "tokenizer"])

        self.model = model
        self.tokenizer = tokenizer
        self.reward = reward
        self.reward_buffer = reward_buffer

        self.get_lr_at_step = lambda step: min(step / 20 * lr, lr)
        self.get_reward_temp_at_step = lambda step: reward_temp_start + (
            reward_temp_end - reward_temp_start
        ) * min(1, step / reward_temp_horizon)

        # end step generation at newline char
        self.end_of_step_token_id = tokenizer.encode("assumption\n", add_special_tokens=False)[-1]

        # TODO: add a method to format the prompt into the format seen in PT/FT
        self.format_prompt = lambda goal_str: goal_str
    

    def forward(
        self, 
        theorem: Theorem, 
        n_samples: Optional[int | list[int]] = None, 
        pf_temperature: float =  1.0, 
        max_depth: int = 3, # depth starts at 0
        replay_tactics: Optional[ProofTreeNode] = None,
        generate_func: Optional[callable] = None,
    ) -> tuple[ProofTreeNode, torch.Tensor, torch.Tensor]:
        """
        Samples a proof tree for a given theorem
        Returns
            - root: the root of the proof tree (ProofTreeNode)
            - trajectories_logpf: (n_samples, max_depth) tensor of log probabilities of each tactic
            - log_r: (n_samples,) tensor of log rewards

        """
        trajectories_logpf = []
        leaf_nodes = []
        log_reward = []
        n_samples = [n_samples or self.hparams.n_samples] * max_depth
        with lean_context(theorem, replay_tactics) as (dojo, root):
            trajectory_logpf = []
            stack = [(root, False)]
            while stack:
                node, visited = stack.pop()
                # node should not be terminal
                # assert node.depth < max_depth and isinstance(node, TacticState)
                if visited:
                    # backing out; clean up this node's element from the trajectory
                    if node.tactic_pf is not None:
                        trajectory_logpf.pop()
                    continue

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
        
        # log_pf is potentially jagged; may need padding
        # log_pf = torch.stack(log_pf, dim=0)
        trajectories_logpf = torch.nn.utils.rnn.pad_sequence(
            trajectories_logpf, batch_first=True, padding_value=0.0
        )
        if replay_tactics:
            log_reward = torch.tensor(log_reward)
        else:
            # compute and assign log_r to leaf nodes
            trajectories = extract_trajectories(root, theorem.uid)
            states = [t["states"] for t in trajectories]
            tactics = [t["tactics"] for t in trajectories]
            log_reward = self.reward(states, tactics)
            for leaf_node, log_r in zip(leaf_nodes, log_reward):
                leaf_node.log_r = log_r.item()

        return root, trajectories_logpf, log_reward

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
        # Should always be (1, prompt_len)
        theorem_id = theorem.uid
        if (
            random.random() < self.hparams.use_buffer_prob
            and (replay_tree := self.reward_buffer.sample(theorem_id, self.hparams.n_samples))
        ):
            # Using a sample from the reward buffer
            root, trajectories_logpf, log_r = self.forward(theorem, replay_tactics=replay_tree)
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
            root, trajectories_logpf, log_r = self.forward(theorem, pf_temperature=pf_temp)
            self.reward_buffer.add_batch(extract_trajectories(root, theorem_id))

        # Get the GFN loss
        # SubTB requires estimating flow (could be implemented as a scalar head over the verifier)
        # - for the proof of concept, we'll just use vanilla TB
        loss = self.tb_loss(log_pf=trajectories_logpf, log_r=log_r)
        # loss = modified_subtb_loss(
        #     log_pf=log_pf,
        #     log_r=log_r,
        #     log_pterm=log_pterm,
        #     generated_text=generated_text,
        #     termination_token_id=self.end_of_sentence_token_id,
        #     prompt_len=len(prompt),
        #     subtb_lambda=self.hparams.subtb_lambda,
        # )

        # Log metrics
        # _, last_log_r, last_log_r_unpenalized, sentence_len = get_termination_vals(
        #     generated_text=generated_text,
        #     log_pf=log_pf,
        #     log_pterm=log_pterm,
        #     log_r=log_r,
        #     log_r_unpenalized=log_r_unpenalized,
        #     termination_token_id=self.end_of_sentence_token_id,
        #     prompt_len=len(prompt),
        # )
        # log_ps = last_log_r * self.reward.temperature
        # log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature
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
        # self.log(
        #     "train/logP(s) (avg)",
        #     log_ps.mean(),
        #     on_step=False,
        #     on_epoch=True,
        #     sync_dist=True,
        # )
        # self.log(
        #     "train/logP(s) (max)",
        #     log_ps.max(),
        #     on_step=False,
        #     on_epoch=True,
        #     sync_dist=True,
        # )
        # self.log(
        #     "train/logP(s) unpenalized (avg)",
        #     log_ps_unpenalized.mean(),
        #     on_step=False,
        #     on_epoch=True,
        #     sync_dist=True,
        # )
        # self.log(
        #     "train/logP(s) unpenalized (max)",
        #     log_ps_unpenalized.max(),
        #     on_step=False,
        #     on_epoch=True,
        #     sync_dist=True,
        # )
        # self.log(
        #     "train/sentence_len",
        #     sentence_len.float().mean(),
        #     on_step=False,
        #     on_epoch=True,
        #     sync_dist=True,
        # )

        return loss

    def validation_step(self, theorem: Theorem, batch_idx: int):
        # sample a proof and get the reward
        _, log_pf, log_r = self.forward(theorem)

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
        # if self.diversity_metric.method is not None:
        #     generated_sentences = self.tokenizer.batch_decode(
        #         generated_text[:, len(prompt) :]
        #     )
        #     generated_sentences = [
        #         text.replace(".", "") for text in generated_sentences
        
        #     diversity = self.diversity_metric(generated_sentences)
        #     self.log(f"val/{self.diversity_metric_name}", diversity, sync_dist=True)

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
                root, log_pf, log_r = self.forward(probe, n_samples=n_samples)
            trajectories = extract_trajectories(root, probe.uid)
            log_pfs = log_pf.sum(dim=-1)
            for i, trajectory in enumerate(trajectories):
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
            root, trajectories_logpf, log_r = self.forward(
                theorem, 
                n_samples=n_samples,
                generate_func=generation_func,
            )
            trajectories = extract_trajectories(root, theorem.uid)
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
                for i, trajectory in enumerate(trajectories)
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
        If replay is True, the method recomputes the log_pf of the node's children instead.

        """
        assert (lean_env is not None) or replay, "lean_env must be provided if not replaying tactics"
        if (node.state and not isinstance(node.state, TacticState)) or node.depth >= max_depth:
            return
        n_samples = n_samples or self.hparams.n_samples
        prompt_text = self.format_prompt(node.state.pp if node.state else node.state_str) 
        prompt = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        if prompt.ndim == 1:
            prompt = prompt.unsqueeze(0)
        prompt = prompt.expand(n_samples, -1)
        reward_fn = partial(
            self.reward.score,
            prompt_length=prompt.shape[1],
            model=self.model,
            tokenizer=self.tokenizer,
        )
        # sample tactics (if not provided by action_seq) and compute terms needed for loss
        if generate_func:
            outputs = generate_func(prompt)
        else:
            tokens, log_pf = generate_step(
                self.model,
                prompt,
                reward_fn=reward_fn,
                termination_token_id=self.end_of_sentence_token_id,
                vocab_naughty_mask=self.hparams.illegal_token_mask,
                min_len=self.hparams.min_sentence_len,
                max_len=self.hparams.max_sentence_len,
                temperature=pf_temperature,
                skip_rewards=False,
                action_seq=node.next_tactic_token_ids if replay else None,
            )
        # log_pf outputted by generate_step is at token-level granularity
        log_pf_tactic = log_pf.sum(dim=1)
        
        if replay:
            for i, child in enumerate(node.children):
                child.step_log_pf = log_pf_tactic[i].item()
                child.trajectory_logpf = torch.cat(node.trajectory_logpf + [log_pf_tactic[i].item()])
        else:
            # save the generated tactics for replay
            node.next_tactic_token_ids = tokens[:, prompt.shape[1]:]
            # pass the generated tactics through the lean environment
            generated_tactics = self.tokenizer.batch_decode(
                tokens[:, prompt.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            tactics_token_length = (tokens[:, prompt.shape[1]:] == self.end_of_sentence_token_id).count_nonzero(dim=-1)
            state_action_length = prompt.shape[1] + tactics_token_length
            # create new child nodes by running the 
            for i, tactic in enumerate(generated_tactics):
                next_state = lean_env.run_tac(node.state, tactic.rstrip())
                child_node = ProofTreeNode(
                    state=next_state,
                    tactic=generated_tactics[i],
                    depth=node.depth + 1,
                    prompt_length=prompt.shape[1],
                    step_log_pf=log_pf_tactic[i],
                    token_tensor= tokens[i, :state_action_length[i]],
                    parent=node,
                    # trajectory_logpf=torch.cat(node.trajectory_logpf + [log_pf_tactic[i].item()]),
                    # - from template:
                    # log_pf=log_pf,
                    # log_pterm=log_pterm,
                    # log_r=log_r,
                    # log_r_unpenalized=log_r_unpenalized,
                )
                if node.children is None:
                    node.children = []
                node.children.append(child_node)


# helper routine
def generate_step(
    model,
    encoded_prompt,
    termination_token_id,
    reward_fn,
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    vocab_alpha=-99,
    max_len=10,
    min_len=0,
    temperature=1.0,
    top_k=999999,
    top_p=1.0,
    action_seq=None,
    skip_rewards=False,
):
    # generate and return the probability of terminating at every step
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    state = encoded_prompt.clone()
    log_pf = []
    # NOTE:
    # - original next sentence log_pterm code is commented out
    # - currently stubbing it with a tensor of zeros (e.g. p(term) = 1)
    # - this is to marginally improve efficiency without changing return signature
    # log_pterm = []
    token_ids = state  # For caching hidden states during generation
    past_key_values = None  # For caching hidden states during generation
    for i in range(max_len + 1):
        output = model(input_ids=token_ids, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]
        if action_seq is None:
            with torch.no_grad():
                prob = logits.softmax(dim=-1)
                modified_logits = logits.clone().detach()
                # implement top-k by getting the top-k largest values and setting the rest to 0
                if top_k < 999999:
                    modified_logits[prob >= prob.topk(top_k)] = -torch.inf
                # implement top-p by getting indices in the top-p prob mass and setting the rest to 0
                if top_p < 1.0:
                    sorted_probs, _ = torch.sort(prob, dim=-1, descending=True)
                    cumsum_prob = torch.cumsum(sorted_probs, dim=-1)
                    nucleus = cumsum_prob < top_p
                    nucleus = torch.cat(
                        [
                            nucleus.new_ones(nucleus.shape[:-1] + (1,)),
                            nucleus[..., :-1],
                        ],
                        dim=-1,
                    )
                    modified_logits[~nucleus] = -torch.inf
                if i < min_len:
                    # if we haven't reach the minimum length, set the probability of terminating to 0
                    modified_logits[:, termination_token_id] = -torch.inf
                elif i >= max_len:
                    # if we've reached the maximum length, set the probability of terminating to 1
                    mask = [True] * modified_logits.shape[1]
                    mask[termination_token_id] = False
                    modified_logits[:, mask] = -torch.inf
                # penalize non-'nice' and 'naughty' tokens with vocab_alpha
                if vocab_nice_mask is not None:
                    modified_logits[:, ~vocab_nice_mask] += vocab_alpha
                if vocab_naughty_mask is not None:
                    modified_logits[:, vocab_naughty_mask] += vocab_alpha
                prob = (modified_logits / temperature).softmax(dim=-1)
                token_ids = torch.multinomial(prob, num_samples=1)
        else:
            if i >= action_seq.size(-1):
                # end with termination token
                token_ids = (
                    torch.ones_like(action_seq[:, 0]) * termination_token_id
                ).unsqueeze(-1)
            else:
                token_ids = action_seq[:, i].unsqueeze(-1)
        token_ids = torch.where(
            active_seqs.unsqueeze(-1),
            token_ids,
            termination_token_id,
        )
        if vocab_nice_mask is not None:
            logits[:, ~vocab_nice_mask] += vocab_alpha
        if vocab_naughty_mask is not None:
            logits[:, vocab_naughty_mask] += vocab_alpha
        # logprob has ndim=2 and shape (batch_size, vocab_size)
        logprob = logits.log_softmax(dim=-1)
        # log_pterm.append(
        #     torch.where(
        #         active_seqs,
        #         logprob[:, termination_token_id],
        #         0,
        #     )
        # )
        active_seqs = active_seqs * (token_ids != termination_token_id).squeeze(-1)
        log_pf.append(
            torch.where(
                active_seqs,
                logprob.gather(-1, token_ids).squeeze(-1),
                0,
            )
        )
        state = torch.cat([state, token_ids], dim=-1)
        # check if all sequences have terminated
        if torch.all(~active_seqs):
            break
    # log_pf does not include the initial prompt
    # before this stack operation, it is a list of tensors of shape (batch_size,)
    # after this stack operation, it is a tensor of shape (batch_size, max_completion_len)
    log_pf = torch.stack(log_pf, dim=1)
    # log_pterm = torch.stack(log_pterm, dim=1)
    # log_pterm = torch.zeros_like(log_pf)
    if skip_rewards:
        log_r, log_r_unpenalized = None, None
    else:
        # Reward for all intermediate states
        # - except the last one (guaranteed to be the termination token)
        log_r, log_r_unpenalized = reward_fn(state[:, :-1])
    # add a termination token to the end of the sequence
    # return state, log_pf, log_pterm, log_r, log_r_unpenalized
    return state, log_pf

@contextmanager
def lean_context(theorem: Theorem, replay_tactics: Optional[ProofTreeNode] = None):
    if replay_tactics:
        yield None, replay_tactics
    else:
        with Dojo(theorem) as (dojo, initial_state):
            new_root = ProofTreeNode(state=initial_state, children=[], trajectory_logpf=[])
            yield dojo, new_root
