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
from proof_tree import ProofTreeNode
from collections import deque


class NeuralTheoremProvingTask(LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        reward,
        reward_buffer,
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
    
    def expand_node(
        self,
        node: ProofTreeNode,
        n_samples: Optional[int] = None,
        pf_temperature: float = 1.0,
        max_depth: int = 3,
        lean_env: Optional[Dojo] = None,
        replay: bool = False,
    ) -> None:
        # generates children for proof tree node
        # - returns number of leaves generated
        assert (lean_env is not None) or replay, "lean_env must be provided if not replaying tactics"
        if not isinstance(node.state, TacticState):
            return
        n_samples = n_samples or self.hparams.n_samples
        prompt_text = self.format_prompt(node.state.pp)
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
            # create new child nodes by running the 
            for i, tactic in enumerate(generated_tactics):
                next_state = lean_env.run_tac(node.state, tactic.rstrip())
                child_node = ProofTreeNode(
                    state=next_state,
                    tactic=generated_tactics[i],
                    depth=node.depth + 1,
                    prompt_length=prompt.shape[1],
                    step_log_pf=log_pf_tactic[i],
                    trajectory_logpf=torch.cat(node.trajectory_logpf + [log_pf_tactic[i].item()]),
                    # - from template:
                    # log_pf=log_pf,
                    # log_pterm=log_pterm,
                    # log_r=log_r,
                    # log_r_unpenalized=log_r_unpenalized,
                )
                node.children.append(child_node)

    def forward(
        self, 
        theorem: Theorem, 
        n_samples: Optional[int | list[int]] = None, 
        pf_temperature: float =  1.0, 
        max_depth: int = 3, # depth starts at 0
        replay_tactics: Optional[ProofTreeNode] = None,
    ):
        log_pf = []
        n_samples = [n_samples or self.hparams.n_samples] * max_depth
        with lean_context(theorem, replay_tactics) as (dojo, root):
            queue = deque([root])
            while queue:
                node = queue.popleft()
                # terminal nodes require no further action
                if not isinstance(node.state, TacticState):
                    continue
                self.expand_node(
                    node, 
                    n_samples = n_samples[node.depth], 
                    pf_temperature=pf_temperature,
                    max_depth=max_depth,
                    lean_env=dojo,
                    replay=(replay_tactics is not None),
                )
                # TODO: add logic to update log_pf as needed here
                # only expand children if (1) above max depth (2) child is non-terminal node
                if node.depth + 1 < max_depth:
                    queue.extend([c for c in node.children if isinstance(c.state, TacticState)])
        
        # log_pf is potentially jagged; may need padding
        # log_pf = torch.stack(log_pf, dim=0)
        log_pf = torch.nn.utils.rnn.pad_sequence(log_pf, batch_first=True, padding_value=0.0)
        return log_pf, root
        

    def training_step(self, prompt, batch_idx):
        # Should always be (1, prompt_len)
        prompt = prompt[0]

        # Sample a sentence and get the reward
        if (
            random.random() < self.hparams.use_buffer_prob
            and self.reward_buffer.sample(self.hparams.n_samples, prompt)[0] is not None
        ):
            # Using a sample from the reward buffer
            action_seq, log_r = self.reward_buffer.sample(
                self.hparams.n_samples, prompt
            )
            generated_text, log_pf, log_pterm, _, log_r_unpenalized = self.forward(
                prompt, action_seq=action_seq
            )
            log_r = log_r[
                :, : generated_text.shape[1] - len(prompt)
            ]  # Undo padding from buffer
            log_r *= 1 / self.reward.temperature  # redo the effect of reward tempering
        else:
            # Using the forward policy
            if random.random() < self.hparams.pf_temp_prob:  # With tempering
                pf_temp = (
                    random.random()
                    * (self.hparams.pf_temp_high - self.hparams.pf_temp_low)
                    + self.hparams.pf_temp_low
                )
            else:  # Without tempering
                pf_temp = 1.0
            generated_text, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
                prompt, pf_temperature=pf_temp
            )
            self.reward_buffer.add_batch(
                prompt=prompt,
                sentences=generated_text[:, len(prompt) :],
                logrewards=log_r
                * self.reward.temperature,  # undo the effect of reward tempering
                tokenizer=self.tokenizer,
            )

        # Get the GFN loss
        loss = modified_subtb_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(prompt),
            subtb_lambda=self.hparams.subtb_lambda,
        )

        # Log metrics
        _, last_log_r, last_log_r_unpenalized, sentence_len = get_termination_vals(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(prompt),
        )
        log_ps = last_log_r * self.reward.temperature
        log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature
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
            last_log_r.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) (avg)",
            log_ps.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) (max)",
            log_ps.max(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) unpenalized (avg)",
            log_ps_unpenalized.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/logP(s) unpenalized (max)",
            log_ps_unpenalized.max(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/sentence_len",
            sentence_len.float().mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, prompt, batch_idx):
        # Should always be (1, prompt_len)
        prompt = prompt[0]

        # Sample a sentence and get the reward
        generated_text, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
            prompt
        )

        # Get the GFN loss
        loss = modified_subtb_loss(
            log_pf=log_pf,
            log_r=log_r,
            log_pterm=log_pterm,
            generated_text=generated_text,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(prompt),
            subtb_lambda=self.hparams.subtb_lambda,
        )

        # Log metrics
        _, last_log_r, last_log_r_unpenalized, sentence_len = get_termination_vals(
            generated_text=generated_text,
            log_pf=log_pf,
            log_pterm=log_pterm,
            log_r=log_r,
            log_r_unpenalized=log_r_unpenalized,
            termination_token_id=self.end_of_sentence_token_id,
            prompt_len=len(prompt),
        )
        log_ps = last_log_r * self.reward.temperature
        log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val/logR",
            last_log_r.mean(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) (avg)",
            log_ps.mean(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) (max)",
            log_ps.max(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) unpenalized (avg)",
            log_ps_unpenalized.mean(),
            sync_dist=True,
        )
        self.log(
            "val/logP(s) unpenalized (max)",
            log_ps_unpenalized.max(),
            sync_dist=True,
        )
        self.log(
            "val/sentence_len",
            sentence_len.float().mean(),
            sync_dist=True,
        )
        if self.diversity_metric.method is not None:
            generated_sentences = self.tokenizer.batch_decode(
                generated_text[:, len(prompt) :]
            )
            generated_sentences = [
                text.replace(".", "") for text in generated_sentences
            ]
            diversity = self.diversity_metric(generated_sentences)
            self.log(f"val/{self.diversity_metric_name}", diversity, sync_dist=True)

    def on_train_batch_start(self, prompt, batch_idx):
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
        # Log variance of (logR - logP(s)) using exploration, which should be 0.0
        log_rs, log_pfss = [], []
        val_data = self.trainer.datamodule.val_dataloader().dataset
        for prompt in val_data:
            prompt = prompt[0]
            generated_text, log_pf, log_pterm, log_r, log_r_unpenalized = self.forward(
                prompt.to(self.device), pf_temperature=2.0
            )
            log_pfs, log_r, _, _ = get_termination_vals(
                generated_text=generated_text,
                log_pf=log_pf,
                log_pterm=log_pterm,
                log_r=log_r,
                log_r_unpenalized=log_r_unpenalized,
                termination_token_id=self.end_of_sentence_token_id,
                prompt_len=len(prompt),
            )
            log_rs.append(log_r)
            log_pfss.append(log_pfs)
        log_rs, log_pfss = torch.cat(log_rs), torch.cat(log_pfss)
        self.log("val/Var(logR - logPf(s))", (log_rs - log_pfss).var(), sync_dist=True)

        # Log probe samples
        if (
            self.hparams.val_probes is not None
            and self.logger is not None
            and self.trainer.current_epoch % 5 == 0
        ):
            samples_table = self.sample_probes(self.hparams.val_probes)
            self.logger.log_table("samples/val_probes", dataframe=samples_table)

    def on_train_start(self):
        # Log baseline metrics
        val_data = self.trainer.datamodule.val_dataloader().dataset
        baseline_performance = None
        for prompt in val_data:
            prompt = prompt[0]
            samples = self.sample_baselines(
                prompt.to(self.device), n_samples=self.hparams.n_samples
            )
            if baseline_performance is None:
                baseline_performance = pd.DataFrame(
                    data=np.zeros((6, len(samples))),
                    columns=samples.keys(),
                    index=[
                        "logP(s) (avg)",
                        "logP(s) (max)",
                        "logP(s) unpenalized (avg)",
                        "logP(s) unpenalized (max)",
                        self.diversity_metric_name,
                        "sentence length",
                    ],
                )
            for baseline in samples:
                baseline_performance.loc["logP(s) (avg)", baseline] += samples[
                    baseline
                ]["logP(s)"].mean().item() / len(val_data)
                baseline_performance.loc["logP(s) (max)", baseline] += samples[
                    baseline
                ]["logP(s)"].max().item() / len(val_data)
                baseline_performance.loc[
                    "logP(s) unpenalized (avg)", baseline
                ] += samples[baseline]["logP(s) unpenalized"].mean().item() / len(
                    val_data
                )
                baseline_performance.loc[
                    "logP(s) unpenalized (max)", baseline
                ] += samples[baseline]["logP(s) unpenalized"].max().item() / len(
                    val_data
                )
                if samples[baseline][self.diversity_metric_name] is None:
                    baseline_performance.loc[
                        self.diversity_metric_name, baseline
                    ] = None
                else:
                    baseline_performance.loc[
                        self.diversity_metric_name, baseline
                    ] += samples[baseline][self.diversity_metric_name] / len(val_data)
                baseline_performance.loc["sentence length", baseline] += samples[
                    baseline
                ]["sentence length"].float().mean().item() / len(val_data)
        baseline_performance = baseline_performance.reset_index(names="metric")
        if self.logger is not None:
            self.logger.log_table(
                "val/baseline performance", dataframe=baseline_performance
            )

        # Log baseline probes
        if self.hparams.val_probes is not None and self.logger is not None:
            samples_table = self.sample_probes_baselines(self.hparams.val_probes)
            self.logger.log_table(
                "samples/val_probes (baselines)", dataframe=samples_table
            )

    def sample_probes(self, probes, n_samples=4):
        assert isinstance(probes, list) and probes[0].ndim == 1
        samples = []
        for probe in probes:
            probe_str = self.tokenizer.decode(probe)
            with torch.no_grad():
                generated_text, _, _, log_r, log_r_unpenalized = self.forward(
                    probe.to(self.device), n_samples=n_samples
                )
            log_ps, log_ps_unpenalized = get_termination_vals(
                generated_text=generated_text,
                log_pf=None,
                log_pterm=None,
                log_r=log_r,
                log_r_unpenalized=log_r_unpenalized,
                termination_token_id=self.end_of_sentence_token_id,
                prompt_len=len(probe),
            )[1:3]
            log_ps *= self.reward.temperature
            log_ps_unpenalized *= self.reward.temperature
            generated_text = generated_text[:, len(probe) :]
            generated_text = self.tokenizer.batch_decode(generated_text)
            generated_text = [text.replace(".", "") for text in generated_text]
            for i in range(len(generated_text)):
                samples.append(
                    {
                        "Prompt": probe_str,
                        "Sampled sentence": generated_text[i],
                        "logP(s)": log_ps[i].item(),
                        "logP(s) unpenalized": log_ps_unpenalized[i].item(),
                    }
                )
        samples = pd.DataFrame(samples)
        samples = samples.sort_values(by=["Prompt", "logP(s)"], ascending=False)
        return samples

    def sample_probes_baselines(self, probes, n_samples=4):
        assert isinstance(probes, list) and probes[0].ndim == 1
        samples = []
        for probe in probes:
            probe_str = self.tokenizer.decode(probe)
            probe_samples = self.sample_baselines(
                probe.to(self.device), n_samples=n_samples
            )
            for i in range(n_samples):
                sample = {"Prompt": probe_str}
                for baseline in probe_samples:
                    sample[f"Sampled sentence ({baseline})"] = probe_samples[baseline][
                        "sample"
                    ][i]
                    sample[f"logP(s) ({baseline})"] = probe_samples[baseline][
                        "logP(s)"
                    ][i].item()
                    sample[f"logP(s) unpenalized ({baseline})"] = probe_samples[
                        baseline
                    ]["logP(s) unpenalized"][i].item()
                samples.append(sample)

        samples = pd.DataFrame(samples)
        samples = samples.sort_values(by=["Prompt"], ascending=False)

        return samples

    def sample_baselines(self, prompt, n_samples=4):
        # https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationMixin.generate
        # https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationConfig
        assert prompt.ndim == 1
        prompt = prompt.unsqueeze(0)

        def generate(prompt, **kwargs):
            with torch.no_grad():
                lora_to_base(self.model)
                generated_text = self.model.generate(
                    prompt,
                    min_new_tokens=self.hparams.min_sentence_len,
                    max_new_tokens=self.hparams.max_sentence_len + 1,
                    eos_token_id=self.end_of_sentence_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    forced_eos_token_id=self.end_of_sentence_token_id,
                    suppress_tokens=torch.from_numpy(self.hparams.illegal_token_mask)
                    .nonzero()
                    .squeeze(-1),
                    **kwargs,
                )
                base_to_lora(self.model)

                log_r, log_r_unpenalized = self.reward.score(
                    generated_text,
                    prompt_length=prompt.shape[1],
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
                (
                    _,
                    last_log_r,
                    last_log_r_unpenalized,
                    sentence_len,
                ) = get_termination_vals(
                    generated_text=generated_text,
                    log_pf=None,
                    log_pterm=None,
                    log_r=log_r,
                    log_r_unpenalized=log_r_unpenalized,
                    termination_token_id=self.end_of_sentence_token_id,
                    prompt_len=prompt.shape[1],
                )
                log_ps = last_log_r * self.reward.temperature
                log_ps_unpenalized = last_log_r_unpenalized * self.reward.temperature

            generated_text = generated_text[:, prompt.shape[1] :]
            generated_text = torch.where(
                generated_text == self.tokenizer.eos_token_id,
                self.end_of_sentence_token_id,
                generated_text,
            )
            generated_text = self.tokenizer.batch_decode(generated_text)
            generated_text = [text.replace(".", "") for text in generated_text]

            if len(generated_text) > 1:
                diversity = self.diversity_metric(generated_text)
            else:
                diversity = None

            if len(generated_text) == 1:
                generated_text = generated_text * n_samples
                log_ps = log_ps.expand(n_samples, -1)
                log_ps_unpenalized = log_ps_unpenalized.expand(n_samples, -1)

            return {
                "sample": generated_text,
                "logP(s)": log_ps,
                "logP(s) unpenalized": log_ps_unpenalized,
                "sentence length": sentence_len,
                self.diversity_metric_name: diversity,
            }

        samples = {}

        # Beam search
        samples["beam"] = generate(
            prompt=prompt,
            do_sample=False,
            num_beams=n_samples * 5,
            length_penalty=0.0,
        )
        samples["beam [fair]"] = generate(
            prompt=prompt,
            do_sample=False,
            num_beams=n_samples,
            length_penalty=0.0,
        )

        # Diverse beam search
        samples["diverse beam"] = generate(
            prompt=prompt,
            num_beams=n_samples * 5,
            num_beam_groups=n_samples,
            num_return_sequences=n_samples,
            diversity_penalty=1.0,
            length_penalty=0.0,
        )
        samples["diverse beam [fair]"] = generate(
            prompt=prompt,
            num_beams=n_samples,
            num_beam_groups=n_samples,
            num_return_sequences=n_samples,
            diversity_penalty=1.0,
            length_penalty=0.0,
        )

        # Nucleaus sampling
        samples["nucleus"] = generate(
            prompt=prompt,
            do_sample=True,
            num_return_sequences=n_samples,
            top_k=0,
            top_p=0.95,
        )

        # LM
        samples["LM"] = generate(
            prompt=prompt,
            do_sample=True,
            num_return_sequences=n_samples,
            top_k=0,
        )

        # LM with temperature
        samples["LM tempered"] = generate(
            prompt=prompt,
            do_sample=True,
            num_return_sequences=n_samples,
            top_k=0,
            temperature=self.hparams.reward_temp_end,
        )

        # Greedy
        samples["greedy"] = generate(
            prompt=prompt,
            do_sample=False,
        )

        return samples

    def configure_optimizers(self):
        if self.hparams.use_4bit:
            import bitsandbytes as bnb  # fmt: skip
            return bnb.optim.PagedAdamW8bit(self.model.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
    
    def determine_branching_factor(self, n_samples, depth):
        if isinstance(n_samples, int):
            return n_samples
        elif isinstance(n_samples, list):
            return n_samples[depth]
        else:
            return self.hparams.n_samples


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
    log_pterm = torch.zeros_like(log_pf)
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