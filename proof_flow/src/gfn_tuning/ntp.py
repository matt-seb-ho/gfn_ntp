import random
from typing import Optional

import torch
from icecream import ic
from loguru import logger
from peft import PeftModel, PeftModelForCausalLM
from pytorch_lightning import LightningModule
from transformers import (
    AutoTokenizer, 
    DynamicCache, 
    QuantizedCache,
    QuantizedCacheConfig,
)
from torch.nn.utils.rnn import pad_sequence

from proof_flow.src.constants import GFN_POLICY_ADAPTER_NAME
from proof_flow.src.gfn_tuning.proof_tree import (
    ProofTreeNode, extract_trajectories
)
from proof_flow.src.gfn_tuning.replay_buffer import ReplayBuffer
from proof_flow.src.gfn_tuning.reward import NTPReward
from proof_flow.src.prompts import (
    INSTRUCTION_PROMPT_TEMPLATE,
    DEEPSEEK_RM_ST_PROMPT_TEMPLATE_V2,
)
from proof_flow.src.search.proof_search import Status, DistributedProver
from proof_flow.src.utils import (
    CUSTOM_LOG_LEVEL,
    SearchEvalConfig,
    prepare_environment_for_lean_dojo,
    batch_iterator_zip,
    repo_root,
)


prepare_environment_for_lean_dojo()
from lean_dojo import ( # isort: skip
    Dojo,
    DojoCrashError,
    DojoInitError,
    DojoTacticTimeoutError,
    LeanGitRepo,
    TacticState,
    Theorem,
)


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
        model_inference_batch_size: int = 4,
        dojo_timeout: int = 600, # default comes from LeanDojo
        max_input_length: int = 130,
        branch_only_at_root: bool = True,
        debug_log_level: str = CUSTOM_LOG_LEVEL,
        tac_gen_prompt_template: str = DEEPSEEK_RM_ST_PROMPT_TEMPLATE_V2,
        search_eval_probes: Optional[list[dict]] = None,
        search_eval_cfg: Optional[SearchEvalConfig] = None,
        ckpt_dest: str = "checkpoints",
        save_ckpt_on_val: bool = False,
        sanity_check_probes: int = 1,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "model", 
            "tokenizer", 
            "reward", 
            "reward_buffer",
            "search_eval_cfg",
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
        
        # proof search evaluation configuration
        self.search_eval_cfg = search_eval_cfg or SearchEvalConfig()
    

    def forward(
        self, 
        theorem: Theorem, 
        n_samples: Optional[int | list[int]] = None, 
        pf_temperature: float =  1.0, 
        max_depth: int = 3, # depth starts at 0
        generate_func: Optional[callable] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[list]]:
        """
        Samples a proof tree for a given theorem
        Returns
            - trajectories_logpf: (n_samples, max_depth) tensor of log probabilities of each tactic
            - log_r: (n_samples,) tensor of log rewards for complete trajectories
            - extracted_trajectories: list of trajectories for the replay buffer
        """
        trajectories_logpf: list[torch.Tensor] = []

        # n_samples[d] is the number of tactics to sample at depth d
        # if provided as int, we use the same number of samples at each depth
        if not isinstance(n_samples, list):
            if self.hparams.branch_only_at_root:
                n_samples = [n_samples] + ([1] * (max_depth - 1))
            else:
                n_samples = [n_samples or self.hparams.n_samples] * max_depth
        with Dojo(
            theorem, 
            timeout=self.hparams.dojo_timeout
        ) as (dojo, initial_state):
            root = ProofTreeNode(state=initial_state)
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
                    dojo,
                    n_samples=n_samples[node.depth], 
                    pf_temperature=pf_temperature,
                    max_depth=max_depth,
                    generate_func=generate_func,
                    device=self.model_device,
                )
                new_stack_items = []
                if node.children is None or len(node.children) == 0:
                    # node is terminal
                    if len(trajectory_logpf) == 0:
                        # found that root node is terminal, 
                        # no trajectories were generated
                        self._debug_log("root node is terminal.")
                        return None, None, None
                    trajectories_logpf.append(torch.cat(trajectory_logpf))
                else:
                    for child in node.children:
                        if (
                            node.depth + 1 < max_depth 
                            and isinstance(child.state, TacticState)
                        ):
                            new_stack_items.append((child, False))
                        else:
                            # child is terminal
                            trajectories_logpf.append(torch.cat(
                                trajectory_logpf + [child.tactic_logpf]
                            ))
                # add new stack items in reverse order for depth-first traversal
                # - why not iterate reversed(node.children)?
                #   we want to handle the terminal outputs in the correct order
                if new_stack_items:
                    stack.extend(reversed(new_stack_items))
                
        
        # redundant check for no trajectories
        if len(trajectories_logpf) == 0:
            self._debug_log("trajectories_logpf empty after forward pass.")
            return None, None, None

        # trajectories can have different lengths (may be jagged) and need to be padded
        trajectories_logpf = pad_sequence(
            trajectories_logpf, 
            batch_first=True, 
            padding_value=0.0
        )
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

    
    def log_z_variance_loss(
        self, 
        log_pf: torch.Tensor, 
        log_r: torch.Tensor,
    ) -> torch.Tensor:
        # log_pf has shape (batch_size, max_tactic_depth)
        # log_r has shape (batch_size,)
        # https://arxiv.org/pdf/2302.05446
        trajectory_log_pf = log_pf.sum(dim=-1)
        batch_zeta = log_r - trajectory_log_pf
        expectation = batch_zeta.mean()
        loss = ((batch_zeta - expectation) ** 2).sum()
        return loss


    def training_step(
        self, 
        theorem: list[Theorem], 
        batch_idx: int,              # required by PyTorch Lightning(?)
        force_replay: bool = False,  # for testing purposes
    ):
        theorem = theorem[0]
        theorem_id = theorem.uid

        # replay trajectories
        # _sample... helper function handles the logic of whether to use the buffer or not
        # uses buffer if:
        #   (1) random() < hparams.use_buffer_prob 
        #   (2) the theorem has trajectories in the buffer
        replay_ts = self._sample_replay_trajectories(theorem_id, force_replay=force_replay)
        
        if replay_ts is not None:
            # using sample from replay buffer
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

            try:
                t_logpf, log_r, extracted_ts = self.forward(
                    theorem, 
                    pf_temperature=pf_temp
                )
            except (DojoInitError, DojoCrashError) as e:
                self._debug_log(f"train step dojo error: {e}")
                return None
                
            if t_logpf is None:
                # no trajectories were generated
                self._debug_log("forward returned None (0 trajectories generated)")
                return None
            self.reward_buffer.add_batch(theorem_id, extracted_ts)

        # get gfn loss
        # - sub tb requires estimating flow (possible impl: scalar head over RM)
        # - for the proof of concept, we'll just use vanilla TB
        # loss = self.tb_loss(log_pf=t_logpf, log_r=log_r)
        loss = self.log_z_variance_loss(t_logpf, log_r)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=1,
        )
        self.log(
            "train/logR",
            # last_log_r.mean(),
            log_r.mean(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self._debug_log(f"train/loss: {loss.item()}")
        self._debug_log(f"train/logR: {log_r.mean().item()}")
        return loss


    def validation_step(self, theorem: list[Theorem], batch_idx: int):
        # search eval
        if self.hparams.search_eval_probes:
            self.run_proof_search_eval()

        # forward pass on a validation theorem (extremely light search)
        theorem = theorem[0]
        try:
            with torch.no_grad():
                log_pf, log_r, _ = self.forward(theorem)
        except (DojoInitError, DojoCrashError) as e:
            self._debug_log(f"val_step forward hit dojo error: {e}")
            return 
        if log_pf is None:
            # no trajectories generated
            self._debug_log("forward returned None (0 trajectories generated)")
            return

        # get the GFN loss
        # loss = self.tb_loss(log_pf=log_pf, log_r=log_r)
        loss = self.log_z_variance_loss(log_pf=log_pf, log_r=log_r)

        # Log metrics
        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            batch_size=1,
        )
        self.log(
            "val/logR",
            log_r.mean(),
            sync_dist=True,
            batch_size=1,
        )
    

    def run_proof_search_eval(self):
        probes = self.hparams.search_eval_probes
        if self.trainer.sanity_checking:
            if self.hparams.sanity_check_probes == 0:
                return
            probes = probes[:self.hparams.sanity_check_probes]
        self._debug_log(f"starting search eval on step {self.global_step}")
        # get repo and theorems
        repo = LeanGitRepo(probes[0]["url"], probes[0]["commit"])
        thms = [
            Theorem(repo, thm["file_path"], thm["full_name"]) 
            for thm in probes
        ]
        positions = [None] * len(thms) # ignored by HF tac gen
        prover = DistributedProver(
            use_vllm=False, # use_vllm
            gen_ckpt_path="", # gen_ckpt_path (needs to be not None)
            ret_ckpt_path=None, # ret_ckpt_path
            indexed_corpus_path=None, # indexed_corpus_path
            max_inp_seq_len=self.search_eval_cfg.max_input_seq_len,
            max_oup_seq_len=self.search_eval_cfg.max_output_seq_len,
            length_penalty=self.search_eval_cfg.length_penalty,
            tactic=None, # tactic
            module=None, # module
            num_workers=self.search_eval_cfg.num_workers,
            num_gpus=self.search_eval_cfg.num_gpus,
            timeout=self.search_eval_cfg.timeout,
            max_expansions=self.search_eval_cfg.max_expansions,
            max_depth=self.search_eval_cfg.max_depth,
            num_sampled_tactics=self.search_eval_cfg.num_sampled_tactics,
            max_new_tokens=self.search_eval_cfg.max_new_tokens,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_template=self.hparams.tac_gen_prompt_template,
            is_decoder_only=True,
        )
        with torch.no_grad():
            results = prover.search_unordered(repo, thms, positions)
        num_proved = 0
        for r in results:
            if r is not None and r.status == Status.PROVED:
                num_proved += 1
        self._debug_log(f"search eval: {num_proved} proved out of {len(thms)}")
        self.log("val/num_proved", num_proved, sync_dist=True, batch_size=1)
    

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
        # ensure training mode is on
        self.model.train()
    

    def on_validation_epoch_start(self):
        self.model.eval()
    

    def on_validation_epoch_end(self):
        self.model.train()
        # save model
        if self.hparams.save_ckpt_on_val:
            save_dir = repo_root() / self.hparams.ckpt_dest / self.global_step
            save_dir.mkdir(parents=True, exist_ok=True)
            # TODO: make this not depend on a constant
            self.model.save_pretrained(
                save_directory=save_dir,
                selected_adapters=[GFN_POLICY_ADAPTER_NAME],
            )


    def configure_optimizers(self):
        if self.hparams.use_4bit:
            import bitsandbytes as bnb  # fmt: skip
            return bnb.optim.PagedAdamW8bit(self.model.parameters(), lr=self.hparams.lr)
        else:
            return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
    

    def expand_node(
        self,
        node: ProofTreeNode,
        lean_env: Dojo,
        n_samples: Optional[int] = None,
        pf_temperature: float = 1.0,
        max_depth: int = 3,
        generate_func: Optional[callable] = None,
        device: Optional[str | torch.device] = None,
    ) -> None:
        """
        Expands a node in the proof tree by generating tactics.

        """
        if (node.state and not isinstance(node.state, TacticState)) or node.depth >= max_depth:
            return

        # generate n new tactics
        prompt_text = self.format_prompt(node.state.pp if node.state else node.state_str) 
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
        if input_ids.shape[1] > self.hparams.max_input_length:
            # early exit to avoid OOM issues
            return
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        n_samples = n_samples or self.hparams.n_samples
        input_ids = input_ids.expand(n_samples, -1)
        prompt_length = input_ids.shape[1]
        if device:
            input_ids = input_ids.to(device)

        # sample tactics (if not provided by action_seq) and compute terms needed for loss
        if generate_func:
            # for baseline generation/decoding methods
            tokens, log_pf = generate_func(input_ids)
        else:
            tokens, log_pf = generate_step(
                self.model,
                input_ids,
                termination_token_id=self.end_of_step_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=pf_temperature,
                max_len=self.hparams.max_tactic_tokens,
            )

        # log_pf outputted by generate_step is at token-level granularity
        # NOTE: this assumes that padded tokens have a log_pf of 0
        log_pf_tactic = log_pf.sum(dim=1)
        
        # get non-padded length of each sequence in the batch
        # NOTE: this implementation excludes the end of step token 
        #       (eos token omitted in replay buffer's state_tactic_tokens)
        is_end_of_step = (tokens[:, prompt_length:]).eq(self.end_of_step_token_id)
        pre_pad_length = is_end_of_step.float().argmax(dim=-1) + prompt_length

        # create new children by running the generated tactics through lean
        generated_tactics = self.tokenizer.batch_decode(
            tokens[:, input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        self._debug_log(f"generated_tactics: {generated_tactics}")
        for i, tactic in enumerate(generated_tactics):
            try:
                next_state = lean_env.run_tac(node.state, tactic.rstrip())
            except (DojoTacticTimeoutError, DojoCrashError) as e:
                self._debug_log(f"run_tac error ({e}) on tactic: {tactic}")
                next_state = None # this gets converted to timeout
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
    

    def _debug_log(self, msg: str):
        logger.log(self.hparams.debug_log_level, msg)
    

    def _sample_replay_trajectories(
        self, 
        theorem_id: str,
        force_replay: bool = False,
    ) -> Optional[list[dict]]:
        # if force replay is True, this random check is skipped
        if (not force_replay and random.random() >= self.hparams.use_buffer_prob):
            return None
        return self.reward_buffer.sample(theorem_id, self.hparams.n_samples)


    def format_prompt(self, state: str):
        # TODO: verify Dojo's pp output is the "goals:..."
        # prepending "-- " to every line to match the evaluation setup in Llemma-7B paper
        # - see figure 4
        # appending a newline to the end of the prompt if it doesn't exist
        # lines = state.split("\n")
        # lines[-1] = lines[-1].rstrip() + "\n"
        # # TODO: check if we want line.lstrip() 
        # commented_lines = ["-- INPUT:"] + ["-- " + line for line in lines]
        # return "\n".join(commented_lines)
        # return INSTRUCTION_PROMPT_TEMPLATE.format(state=state)
        # return DEEPSEEK_RM_ST_PROMPT_TEMPLATE_V2.format(state=state)
        return self.hparams.tac_gen_prompt_template.format(state=state)
    

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # return the batch as-is, without moving it to the device
        # assuming batch has type list[Theorem]
        return batch


def generate_step_hf(
    model: PeftModel,
    input_ids: torch.Tensor,
    termination_token_id: int,
    temperature: float = 1.0,
    top_k: int = 999999,
    top_p: float = 1.0,
    prompt_length: Optional[int] = None,
    max_new_tokens: int = 30,
    **generation_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    forward pass for a single state
    returns:
    1. token id tensor of (state, tactic)
    2. log probability tensor for the tactics
    """
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


def generate_step(
    model: PeftModelForCausalLM,
    encoded_prompt: torch.Tensor,
    termination_token_id: int,
    pad_token_id: int,
    vocab_nice_mask: Optional[torch.Tensor] = None,
    vocab_naughty_mask: Optional[torch.Tensor] = None,
    vocab_alpha: int = -99,
    max_len: int = 30,
    min_len: int = 1,
    temperature: float = 1.0,
    top_k: int = 999999,
    top_p: float = 1.0,
    use_quantized_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    alternate implementation to HuggingFace's .generate() WITH GRADIENTS
    - TODO: add beam search?

    args (required)
        model
        encoded_prompt: input_ids tensor shape (batch_size, input_seq_len)
        termination_token_id
        pad_token_id
    returns
        token_ids: tensor of sampled tokens with shape (batch_size, max_seq_len)
        log_pf: tensor of log_pf for sampled tokens (batch_size, max_new_tokens)
    """
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    state = encoded_prompt.clone()
    log_pf = []
    token_ids = state  # For caching hidden states during generation
    # past_key_values = None  # For caching hidden states during generation
    if use_quantized_cache:
        kv_cache = QuantizedCache(
            QuantizedCacheConfig(
                compute_dtype=torch.bfloat16,
                device=encoded_prompt.device,
            )
        )
    else:
        kv_cache = DynamicCache()
    for i in range(max_len + 1):
        output = model(
            input_ids=token_ids, 
            past_key_values=kv_cache,
            use_cache=True,
            return_dict=True,
        )
        # past_key_values = output.past_key_values
        kv_cache = output.past_key_values
        logits = output.logits[:, -1, :]
        
        # sample out a token from logits
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
            if vocab_nice_mask is not None:
                # add vocab_alpha to the logits of the unmasked vocab items
                modified_logits[:, ~vocab_nice_mask] += vocab_alpha
            if vocab_naughty_mask is not None:
                # add vocab_alpha to the logits of the masked vocab items
                modified_logits[:, vocab_naughty_mask] += vocab_alpha
            prob = (modified_logits / temperature).softmax(dim=-1)
            token_ids = torch.multinomial(prob, num_samples=1)

        token_ids = torch.where(
            active_seqs.unsqueeze(-1),
            token_ids,
            # termination_token_id,
            pad_token_id,
        )
        if vocab_nice_mask is not None:
            logits[:, ~vocab_nice_mask] += vocab_alpha
        if vocab_naughty_mask is not None:
            logits[:, vocab_naughty_mask] += vocab_alpha
        logprob = logits.log_softmax(dim=-1)
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

    log_pf = torch.stack(log_pf, dim=1)
    return state, log_pf

