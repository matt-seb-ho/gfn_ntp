import json
import random
from collections import namedtuple
from typing import Optional

import torch
from icecream import ic
from loguru import logger
from peft import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
)
from pytorch_lightning import LightningModule
from transformers import (
    AutoTokenizer, 
    DynamicCache, 
    QuantizedCache,
    QuantizedCacheConfig,
)
from torch.nn.utils.rnn import pad_sequence

from proof_flow.src.constants import GFN_POLICY_ADAPTER_NAME, TACTIC_DELIMITER
from proof_flow.src.gfn_tuning.proof_tree import (
    ProofTreeNode, 
    convert_tactic_result_to_state_string,
    extract_trajectories,
)
from proof_flow.src.gfn_tuning.replay_buffer import ReplayBuffer, BufferEntry
from proof_flow.src.gfn_tuning.reward import NTPReward
from proof_flow.src.prompts import (
    DEEPSEEK_RM_ST_PROMPT_TEMPLATE_V2,
    REPROVER_TACGEN_WITH_HISTORY,
)
from proof_flow.src.search.common import ProofSearchParams, _HuggingFaceLM
from proof_flow.src.search.proof_search import (
    DistributedProver,
    Status,
    get_cached_dojo,
)
from proof_flow.src.utils import (
    CUSTOM_LOG_LEVEL,
    batch_iterator_zip,
    causal_conditional_log_prob,
    prepare_environment_for_lean_dojo,
    repo_root,
    seq2seq_conditional_log_prob,
)


prepare_environment_for_lean_dojo()
from lean_dojo import ( # isort: skip
    Dojo,
    DojoCrashError,
    DojoInitError,
    DojoTacticTimeoutError,
    LeanGitRepo,
    TacticState,
    TacticResult,
    Theorem,
)


ParallelForwardStepResult = namedtuple(
    "ParallelForwardStepResult",
    [
        "next_states", 
        "tactic_logpf", 
        "tactics", 
        "idx_map", 
    ]
)


class NeuralTheoremProvingTask(LightningModule):
    def __init__(
        self,
        model: _HuggingFaceLM,
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
        replay_batch_size: int = 4,
        dojo_timeout: int = 600, # default comes from LeanDojo
        max_input_length: int = 640,
        branch_only_at_root: bool = True,
        search_eval_probes: Optional[list[dict]] = None,
        search_eval_params: Optional[ProofSearchParams] = None,
        ckpt_dest: str = "checkpoints",
        save_ckpt_on_val: bool = False,
        sanity_check_probes: int = 1,
        debug_log_level: str = CUSTOM_LOG_LEVEL,
        tac_gen_prompt_template: str = DEEPSEEK_RM_ST_PROMPT_TEMPLATE_V2,
        ground_truth_trajectories: Optional[dict] = None,
        repeats_per_accumulated_batch: int = 1,
        seq2seq: bool = False,
        truncate_state: bool = False,
        conditional_log_z: bool = True,
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "model", 
            "tokenizer", 
            "reward", 
            "reward_buffer",
            "search_eval_cfg",
            "ground_truth_trajectories",
        ])

        self.model = model
        self.tokenizer = tokenizer
        self.reward = reward
        self.reward_buffer = reward_buffer
        self.model_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if conditional_log_z:
            self.log_z = None
            log_z_head_input_size = (
                model.config.d_model
                if seq2seq else
                model.config.hidden_size
            )
            self.log_z_head = torch.nn.Linear(log_z_head_input_size, 1)
            if self.model_device:
                self.log_z_head = self.log_z_head.to(self.model_device)
        else:
            # unconditional log_z (single theorem)
            self.log_z = torch.nn.Parameter(
                torch.tensor(0.0, requires_grad=True, device=self.model_device)
            )
            self.log_z_head = None

        # self.get_lr_at_step = lambda step: min(step / 20 * lr, lr)
        self.get_lr_at_step = lambda step: lr
        self.get_reward_temp_at_step = lambda step: reward_temp_start + (
            reward_temp_end - reward_temp_start
        ) * min(1, step / reward_temp_horizon)

        # end step generation at newline char
        self.end_of_step_token_id = tokenizer.encode("assumption\n", add_special_tokens=False)[-1]
        
        # proof search evaluation configuration
        if search_eval_params is None:
            self._debug_log("search_eval_params is None")
            self.search_eval_params = ProofSearchParams()
        else:
            self.search_eval_params = search_eval_params
        
        # trade ram to remove dojo startup time
        # - maps theorem.uid -> (dojo object, initial state)
        self.dojo_cache = {}

        # for adding ground truth trajectories to online training
        self.ground_truth_trajectories = ground_truth_trajectories
        self.gt_tacs = {
            tuid: t.proof.split(TACTIC_DELIMITER)
            for tuid, t in ground_truth_trajectories.items()
        }
        if self.hparams.seq2seq:
            self.generate_step = generate_step_seq2seq
            self.conditional_log_p = seq2seq_conditional_log_prob
        else:
            self.generate_step = generate_step
            self.conditional_log_p = causal_conditional_log_prob
        
        # tracking state length exceeding max_input_length
        self.max_input_length_exceeded = 0
        self.states_tokenized = 0

        # controlling tokenization in parallel forward
        self.tokenize_max_length = (
            self.hparams.max_input_length
            if self.hparams.truncate_state
            else None
        )

        # metrics
        self.log_on_step = True
        self.log_on_epoch = False

    

   
    def parallel_forward(
        self,
        theorem: Theorem,
        n_samples: Optional[int] = None,
        pf_temperature: float = 1.0,
        max_depth: int = 3,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[list]]:
        """
        generates trajectories in parallel for a given theorem
        - trajectories_logpf: (n_samples, max_depth) tensor of log probabilities of each tactic
        - log_r: (n_samples,) tensor of log rewards for complete trajectories
        - extracted_trajectories: list of trajectories for the replay buffer
        """
        n_samples = n_samples or self.hparams.n_samples

        # use cached dojo if available
        # TODO: add mechanism for clearing cache/releasing resources
        dojo, initial_state = get_cached_dojo(
            self.dojo_cache, 
            theorem,
            self.hparams.dojo_timeout,
        )
        
        active_trajectories = [True] * n_samples
        tactics = [[] for _ in range(n_samples)]
        trajectory_logpf = []
        tactic_states = [[initial_state] for _ in range(n_samples)]
        
        for step_idx in range(max_depth):
            # generate tactics for active sequences
            r = self.parallel_forward_step(
                step_idx,
                n_samples,
                active_trajectories,
                tactic_states,
                tactics,
                pf_temperature,
                dojo,
            )
            if r is None:
                break

            # update states and tactics
            for i, next_state in enumerate(r.next_states):
                ti = r.idx_map[i]
                tactic_states[ti].append(next_state)
                tactics[ti].append(r.tactics[i])
                active_trajectories[ti] = isinstance(next_state, TacticState)

            # update trajectory logpf
            tactics_logpf = torch.zeros(n_samples, device=self.model_device)
            tactics_logpf[r.idx_map] = r.tactic_logpf
            trajectory_logpf.append(tactics_logpf)

            # early exit if all trajectories are inactive
            if not any(active_trajectories):
                break
                
        # log generated tactics
        tac_info = json.dumps({
            "thm": theorem.full_name,
            "epoch": self.current_epoch,
            "step": self.global_step,
            "tactics": tactics,
            "gold": self.gt_tacs[theorem.uid],
        })
        logger.info(f"train_fwd_gen_tacs: {tac_info}")
        
        # combine trajectory logpf tensors into a single tensor
        # currently have a list of tensors of shape (n_samples,)
        # want to concatenate along dim=1 to get (n_samples, max_depth)
        trajectory_logpf = torch.stack(trajectory_logpf, dim=1)
        
        # compute log_r for each trajectory
        # - need to assemble list[list[str]] for states
        state_strings = []
        for i in range(n_samples):
            trajectory_states = [
                convert_tactic_result_to_state_string(s)
                for s in tactic_states[i]
            ]
            state_strings.append(trajectory_states)
        # log_r = self.reward.silly_san_check(
        #     tactics,
        #     self.gt_tacs[theorem.uid],
        #     device=self.model_device,
        # )
        log_r = self.reward.score(
            state_strings,
            tactics,
            batch_size=n_samples,
            device=self.model_device,
        )

        # reformat for replay buffer storage
        trajectories = []
        for i in range(n_samples):
            trajectories.append(BufferEntry(
                log_r=log_r[i].item(),
                proof=TACTIC_DELIMITER.join(tactics[i]),
                states=state_strings[i],
            ))
        return trajectory_logpf, log_r, trajectories
            
        
    def parallel_forward_step(
        self,
        idx: int,
        n_samples: int,
        active_trajectories: list[bool],
        tactic_states: list[list[TacticResult]],
        tactics: list[list[str]],
        pf_temperature: float,
        lean_env: Dojo,
    ) -> Optional[ParallelForwardStepResult]:
        # first construct input_ids
        i2t_idx_map = []
        input_texts = []
        for i in range(n_samples):
            if not active_trajectories[i]:
                continue
            input_text = self.format_prompt(tactic_states[i], tactics[i], idx)
            token_length = len(self.tokenizer.encode(input_text))
            self.states_tokenized += 1
            if token_length > self.hparams.max_input_length:
                self.max_input_length_exceeded += 1
                logger.info("max_input_length exceeded (total incidents: {self.max_input_length_exceeded})")
                if not self.hparams.truncate_state:
                    continue
            input_texts.append(input_text)
            i2t_idx_map.append(i)
        
        # early exit if no active trajectories
        if len(input_texts) == 0:
            return None

        # build input_ids tensor
        batch_enc = self.tokenizer(
            input_texts, 
            return_tensors="pt", 
            truncation=self.hparams.truncate_state,
            max_length=self.tokenize_max_length,
            padding=True,
        )
        if self.model_device:
            batch_enc = batch_enc.to(self.model_device)

        tokens, log_pf = self.generate_step(
            self.model,
            encoded_prompt=batch_enc.input_ids,
            termination_token_id=self.end_of_step_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=pf_temperature,
            max_len=self.hparams.max_tactic_tokens,
            attention_mask=batch_enc.attention_mask,
        )

        # log_pf outputted by generate_step is at token-level granularity
        # NOTE: this assumes that padded tokens have a log_pf of 0
        tactic_logpf = log_pf.sum(dim=1)

        # decode tokens to get tactics
        if self.hparams.seq2seq:
            tactic_tokens = tokens
        else:
            tactic_tokens = tokens[:, batch_enc.input_ids.shape[1]:]
        generated_tactics = self.tokenizer.batch_decode(
            tactic_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        next_states = []
        for i, tactic in enumerate(generated_tactics):
            try:
                prev_state = tactic_states[i2t_idx_map[i]][-1]
                next_state = lean_env.run_tac(prev_state, tactic.rstrip())
            except (DojoTacticTimeoutError, DojoCrashError) as e:
                self._debug_log(
                    f"run_tac error ({e}) on tactic: {tactic}, "
                    f"restarting dojo for {lean_env.entry.full_name}"
                )
                lean_env, initial_state = get_cached_dojo(
                    self.dojo_cache,
                    lean_env.entry,
                    self.hparams.dojo_timeout,
                )
                next_state = None # this gets converted to timeout
            next_states.append(next_state)
        
        # tactics = [t.strip() for t in generated_tactics]
        return ParallelForwardStepResult(
            next_states=next_states, 
            tactic_logpf=tactic_logpf, 
            tactics=generated_tactics,
            idx_map=i2t_idx_map,
        )
                

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
        
        # use cached dojo if available
        # TODO: add mechanism for clearing cache/releasing resources
        if theorem.uid in self.dojo_cache:
            dojo, initial_state = self.dojo_cache[theorem.uid]
        else:
            self._debug_log(f"loading dojo for {theorem.full_name}")
            dojo = Dojo(theorem, timeout=self.hparams.dojo_timeout)
            dojo, initial_state = dojo.__enter__()
            self.dojo_cache[theorem.uid] = (dojo, initial_state)
            
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


    def sft_loss(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_z: torch.Tensor,
    ) -> torch.Tensor:
        return -log_pf[-1].sum()    


    def tb_loss(
        self,
        log_pf: torch.Tensor,
        log_r: torch.Tensor,
        log_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the batch loss using the Trajectory Balance objective

        Args:
            log_pf: log probabilities of each tactic/action. shape=(batch, max_depth)
            log_r: log reward. shape=(batch_size,)

        Returns:
            batch_loss: scalar tensor
        """
        # pseudo sft
        # loss = (log_pf[-1].sum()) ** 2
        # return loss

        loss = (log_pf.sum(dim=-1) + log_z - log_r) ** 2
        batch_loss = loss.mean()
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
        # print("tlogpf in var grad loss:", trajectory_log_pf)
        batch_zeta = log_r - trajectory_log_pf
        expectation = batch_zeta.mean().detach()
        # set expectation to 0 means we regress log_pf -> log_r (unnormalized)
        # - before, we're trying to regress log_pf -> log_r - z
        # expectation = 0
        loss = ((batch_zeta - expectation) ** 2).mean()
        return loss


    def training_step(
        self, 
        theorem: list[Theorem], 
        batch_idx: int,
    ) -> tuple:
        theorem = theorem[0]
        theorem_id = theorem.uid

        # replay trajectories
        # _sample... helper function handles the logic of whether to use the buffer or not
        # uses buffer if:
        #   (1) random() < hparams.use_buffer_prob 
        #   (2) the theorem has trajectories in the buffer
        replay_ts = self._sample_replay_trajectories(theorem_id)
        
        if replay_ts is not None:
            # using sample from replay buffer
            # self._debug_log(f"replaying trajectories for {theorem.full_name}")
            t_logpf, log_r, = self.replay_trajectories(
                replay_ts,
                batch_size=self.hparams.replay_batch_size,
            )
            initial_tac_state = replay_ts[0].states[0]
        else:
            # using the forward policy
            if random.random() < self.hparams.pf_temp_prob:  
                # with tempering
                pf_temp = (
                    random.random()
                    * (self.hparams.pf_temp_high - self.hparams.pf_temp_low)
                    + self.hparams.pf_temp_low
                )
            else:
                # without tempering
                pf_temp = 1.0

            try:
                t_logpf, log_r, trajectories = self.parallel_forward(
                    theorem, 
                    pf_temperature=pf_temp
                )
            except (DojoInitError, DojoCrashError) as e:
                self._debug_log(f"train step dojo error: {e}")
                dojo, initial_state = get_cached_dojo(
                    self.dojo_cache,
                    theorem,
                    self.hparams.dojo_timeout,
                )
                return None

            if t_logpf is None:
                # no trajectories were generated
                self._debug_log("forward returned None (0 trajectories generated)")
                return None
            self.reward_buffer.add_batch(theorem_id, trajectories)
            initial_tac_state = trajectories[0].states[0]

        # for tb_loss: estimate log_z
        # - pseudo sft testing: log_z = 0
        # log_z = torch.zeros(1, dtype=torch.float32, device=self.model_device)
        log_z = self._compute_log_z(initial_tac_state)

        # once per accumulation batch
        if (batch_idx + 1) % self.hparams.repeats_per_accumulated_batch == 0:
            # add ground truth trajectory before computing loss
            if self.ground_truth_trajectories:
                gt_tlpf, gt_lr = self.replay_trajectories(
                    [self.ground_truth_trajectories[theorem_id]],
                    batch_size=1,
                )
                t_logpf = self._append_tensor_and_pad(t_logpf, gt_tlpf)
                log_r = torch.cat([log_r, gt_lr])

        # apply reward temperature
        log_r = log_r / self.reward.temperature

        # get gfn loss
        # - sub tb requires estimating flow (possible impl: scalar head over RM)
        # - for the proof of concept, we'll just use vanilla TB
        logger.info(f"t_logpf: {t_logpf.sum(dim=-1)}, log_r: {log_r}, log_z: {log_z}")
        # loss = self.sft_loss(
        #     log_pf=t_logpf,
        #     log_r=log_r,
        #     log_z=log_z,
        # )
        loss = self.tb_loss(
            log_pf=t_logpf, 
            log_r=log_r, 
            log_z=log_z
        )
        # loss = self.log_z_variance_loss(t_logpf, log_r)
        self.log(
            "train/loss",
            loss,
            on_step=self.log_on_step,
            on_epoch=self.log_on_epoch,
            sync_dist=True,
            prog_bar=True,
            batch_size=1,
        )
        self.log(
            "train/logR",
            # last_log_r.mean(),
            log_r.mean(),
            on_step=self.log_on_step,
            on_epoch=self.log_on_epoch,
            sync_dist=True,
            batch_size=1,
        )
        # wandb log:
        # - log_r, log_pf, log_z (per theorem)
        to_log = [
            ("log_r", log_r.mean()),
            ("log_pf", t_logpf.sum(dim=-1).mean()),
            ("log_z", log_z),
        ]
        for k, v in to_log:
            self.log(
                f"{theorem.full_name}_{k}",
                v,
                on_step=self.log_on_step,
                on_epoch=self.log_on_epoch,
                sync_dist=1,
                batch_size=1,
            )

        self._debug_log(f"train/loss: {loss.item()}")
        self._debug_log(f"train/logR: {log_r.mean().item()}")
        return loss


    def validation_step(self, theorem: list[Theorem], batch_idx: int):
        # forward pass on a validation theorem (extremely light search)
        theorem = theorem[0]
        try:
            with torch.no_grad():
                log_pf, log_r, _ = self.parallel_forward(theorem)
                if theorem.uid not in self.dojo_cache:
                    logger.info(f"val step: key error on {theorem.uid} in dojo_cache")
                    return
                _, initial_state = self.dojo_cache[theorem.uid]
                initial_tac_state = initial_state.pp
                log_z = self._compute_log_z(initial_tac_state)
        except (DojoInitError, DojoCrashError) as e:
            self._debug_log(f"val_step forward hit dojo error: {e}")
            return 
        if log_pf is None:
            # no trajectories generated
            self._debug_log("forward returned None (0 trajectories generated)")
            return

        # get the GFN loss
        # loss = self.tb_loss(log_pf=log_pf, log_r=log_r)
        # loss = self.log_z_variance_loss(log_pf=log_pf, log_r=log_r)
        loss = self.tb_loss(log_pf=log_pf, log_r=log_r, log_z=log_z)

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
    

    def run_proof_search_eval(self) -> list:
        probes = self.hparams.search_eval_probes
        if self._is_sanity_checking():
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
            max_inp_seq_len=self.search_eval_params.max_input_seq_len,
            max_oup_seq_len=self.search_eval_params.max_output_seq_len,
            length_penalty=self.search_eval_params.length_penalty,
            tactic=None, # tactic
            module=None, # module
            num_workers=self.search_eval_params.num_workers,
            num_gpus=self.search_eval_params.num_gpus,
            timeout=self.search_eval_params.timeout,
            max_expansions=self.search_eval_params.max_expansions,
            max_depth=self.search_eval_params.max_depth,
            num_sampled_tactics=self.search_eval_params.num_sampled_tactics,
            max_new_tokens=self.search_eval_params.max_new_tokens,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_template=self.hparams.tac_gen_prompt_template,
            is_decoder_only=(not self.hparams.seq2seq),
            end_of_step_token_id=self.end_of_step_token_id,
            dojo_cache=self.dojo_cache,
        )
        with torch.no_grad():
            results = prover.search_unordered(repo, thms, positions)
        num_proved = 0
        for r in results:
            if r is not None and r.status == Status.PROVED:
                num_proved += 1
        self._debug_log(f"search eval: {num_proved} proved out of {len(thms)}")
        self.log("val/num_proved", num_proved, sync_dist=True, batch_size=1)
        return results
    

    def on_train_batch_start(self, theorem, batch_idx):
        # Update scheduled quantities
        reward_temp = self.get_reward_temp_at_step(self.global_step)
        # lr = self.get_lr_at_step(self.global_step)
        lr = self.hparams.lr
        self.reward.temperature = reward_temp
        for pg in self.optimizers().param_groups:
            pg["lr"] = lr


    def on_train_epoch_start(self):
        # Log scheduled quantities
        self.log("scheduled/R_temperature", self.reward.temperature, sync_dist=True)
        # self.log("scheduled/lr", self.get_lr_at_step(self.global_step), sync_dist=True)
        # ensure training mode is on
        self.model.train()
    

    def on_train_end(self):
        msg = (
            "report on how many times the max_input_length was exceeded:\n"
            f"max_input_length_exceeded: {self.max_input_length_exceeded}\n"
            f"states tokenized: {self.states_tokenized}\n"
            f"proportion: {self.max_input_length_exceeded / self.states_tokenized + 1}"
        )
        logger.info(msg)
    

    def on_validation_epoch_start(self):
        self.model.eval()
        if self.hparams.search_eval_probes:
            self.run_proof_search_eval()


    def on_validation_epoch_end(self):
        self.model.train()
        # save model
        if self.hparams.save_ckpt_on_val:
            save_dir = repo_root() / f"{self.hparams.ckpt_dest}/{self.global_step}"
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
        trajectories: list[BufferEntry],
        batch_size: int,
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
        max_depth = max(len(t.states) for t in trajectories) - 1
        step_logpfs = torch.zeros(
            # (len(trajectories), self.hparams.max_tactics), 
            (len(trajectories), max_depth),
            device=self.model_device
        )
        # eager sum version
        # t_logpfs = torch.zeros(len(trajectories), device=self.model_device)

        # queue up jobs
        prompts = []
        completions = []
        batch_idxs = []
        step_idxs = []
        for b_idx, trajectory in enumerate(trajectories):
            tactics = [
                t.strip() + '\n' 
                for t in trajectory.proof.split(TACTIC_DELIMITER)
            ]
            for s_idx in range(len(tactics)):
                prompt = self.format_prompt(
                    trajectory.states,
                    tactics,
                    s_idx,
                    str_tactic_states=True,
                )
                prompts.append(prompt)
                completions.append(tactics[s_idx])
                batch_idxs.append(b_idx)
                step_idxs.append(s_idx)
        
        # compute tactic log_pfs in batches
        for _prompts, _completions, b_idxs, s_idxs in batch_iterator_zip(
            (prompts, completions, batch_idxs, step_idxs), 
            batch_size=batch_size
        ):
            log_pfs, _ = self.conditional_log_p(
                self.model,
                self.tokenizer,
                _prompts,
                _completions,
                max_input_length=self.hparams.max_input_length,
                device=self.model_device,
            )
            step_logpfs[b_idxs, s_idxs] = log_pfs
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
            [t.log_r for t in trajectories], 
            dtype=step_logpfs.dtype,
            device=self.model_device,
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
        theorem_id: str
    ) -> Optional[list[dict]]:
        # mbt = getattr(self, "max_batch_testing", None)
        # if mbt is not None:
        #     res = self.reward_buffer.sample(theorem_id, self.hparams.n_samples)
        #     return [res[0]] * mbt
        if random.random() < self.hparams.use_buffer_prob:
            return self.reward_buffer.sample(theorem_id, self.hparams.n_samples)
        return None


    def format_prompt(
        self,
        tactic_states: list[TacticState] | list[str], 
        tactics: list[str],
        idx: int,
        str_tactic_states: bool = False,
    ) -> str:
        tactics = "\n".join(tactics[:idx])
        if str_tactic_states:
            initial_state = tactic_states[0]
            current_state = tactic_states[idx]
        else:
            initial_state = tactic_states[0].pp
            current_state = tactic_states[idx].pp
        prompt = REPROVER_TACGEN_WITH_HISTORY.format(
            initial_state=initial_state,
            tactics=tactics,
            current_state=current_state,
        )
        return prompt
    

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # return the batch as-is, without moving it to the device
        # assuming batch has type list[Theorem]
        return batch

    
    def _is_sanity_checking(self) -> bool:
        try:
            return self.trainer.sanity_checking
        except RuntimeError:
            # if trainer is not attached, self.trainer will raise a RuntimeError
            # only reason trainer is not attached is that we're in a sanity check
            return True
    

    def _append_tensor_and_pad(
        self, 
        t1: torch.Tensor, 
        t2: torch.Tensor
    ) -> torch.Tensor:
        """
        adds t2 of shape (m,) to t1 of shape (batch_size, n)
        such that resulting tensor has shape (batch_size + 1, max(m, n))
        """
        n = t1.shape[1]
        m = t2.shape[1]
        # match the inner dimension
        if m > n:
            padding = (0, m - n)  # (left_pad, right_pad) along the last dimension
            t1 = torch.nn.functional.pad(t1, padding)
        if n > m:
            t2 = torch.nn.functional.pad(t2, (0, n - m))
        # concatenate along the batch dimension (0-th dimension)
        return torch.cat([t1, t2], dim=0)
    

    def _compute_log_z(self, initial_tac_state: str) -> torch.Tensor:
        if self.hparams.conditional_log_z:
            input_ids = self.tokenizer(
                initial_tac_state, 
                return_tensors="pt"
            ).input_ids
            if self.model_device:
                input_ids = input_ids.to(self.model_device)
            if self.hparams.seq2seq:
                encoder = self.model.get_encoder()
                enc_out = encoder(input_ids)
                # get last hidden state (batch_size, seq_length, hidden_size)
                hidden_states = enc_out.last_hidden_state
                # aggregate hidden states (mean pooling)
                # shape: (batch_size, hidden_size)
                pooled_output = hidden_states.mean(dim=1)
                # pass through the regression head to get scalar output
                # shape: (batch_size, 1)
                log_z = self.log_z_head(pooled_output)
            else:
                output = self.model(
                    input_ids, 
                    output_hidden_states=True,
                    return_dict=True,
                )
                # output.hidden_states is a tuple of tensors (embedding + each layer)
                # each tensor has shape (batch_size, seq_len, hidden_size)
                final_hidden_state = output.hidden_states[-1][:, -1, :]
                log_z = self.log_z_head(final_hidden_state)
        else:
            # unconditional log_z (single theorem)
            log_z = self.log_z
        return log_z


def generate_step(
    model: _HuggingFaceLM,
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
    attention_mask: Optional[torch.Tensor] = None,
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
            attention_mask=attention_mask,
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
            # https://huggingface.co/docs/transformers/main/en/kv_cache#under-the-hood-how-cache-object-works-in-attention-mechanism
            # attn_mask shape SHOULD BE (batch_size, past_kv_length + new_tokens_length)
            attention_mask = torch.cat(
                [
                    attention_mask, 
                    attention_mask.new_ones((attention_mask.shape[0], 1))
                ],
                dim=-1
            )

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


def generate_step_seq2seq(
    model: _HuggingFaceLM,
    encoded_prompt: torch.Tensor,
    termination_token_id: int,
    pad_token_id: int,
    eos_token_id: int,
    vocab_nice_mask: Optional[torch.Tensor] = None,
    vocab_naughty_mask: Optional[torch.Tensor] = None,
    vocab_alpha: int = -99,
    max_len: int = 30,
    min_len: int = 1,
    temperature: float = 1.0,
    top_k: int = 999999,
    top_p: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    simplified implementation of seq2seq LM generation WITH GRADIENTS
    why does the token_ids have shape (batch_size, max_len + 1)?
    - decoder has its own BOS token (decoder_start_token_id)

    args (required)
        model
        encoded_prompt: input_ids tensor shape (batch_size, input_seq_len)
        termination_token_id
        pad_token_id
    returns
        token_ids: tensor of sampled tokens with shape (batch_size, max_len + 1)
        log_pf: tensor of log_pf for sampled tokens (batch_size, max_new_tokens)
    """
    batch_size = encoded_prompt.size(0)
    active_seqs = torch.ones(batch_size).bool().to(encoded_prompt.device)

    # encode prompt
    encoder = model.get_encoder()
    encoder_outputs = encoder(input_ids=encoded_prompt, attention_mask=attention_mask)
    
    # initialize decoder input
    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_input_ids = torch.full(
        (batch_size, 1), 
        decoder_start_token_id, 
        dtype=torch.long, 
        device=encoded_prompt.device
    )

    # https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/t5#transformers.T5ForConditionalGeneration
    # - past_key_values apparently doesn't support cache classes for seq2seq models
    past_key_values = None  # For caching hidden states during generation
    
    # prepare tensor to store log probabilities
    log_pf = torch.zeros(batch_size, max_len + 1, device=encoded_prompt.device)
    
    for i in range(max_len + 1):
        output = model(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            # last token of the decoder input (using cache)
            decoder_input_ids=decoder_input_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = output.past_key_values
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
            next_tokens = torch.multinomial(prob, num_samples=1)

        next_tokens = torch.where(
            active_seqs.unsqueeze(-1),
            next_tokens,
            # termination_token_id,
            pad_token_id,
        )
        if vocab_nice_mask is not None:
            logits[:, ~vocab_nice_mask] += vocab_alpha
        if vocab_naughty_mask is not None:
            logits[:, vocab_naughty_mask] += vocab_alpha
        log_prob_distributions = logits.log_softmax(dim=-1)
        
        # update log_pf before active sequence 
        # so to NOT exclude the termination token's log pf
        log_pf[:, i] = (
            torch.where(
                active_seqs,
                log_prob_distributions.gather(-1, next_tokens).squeeze(-1),
                0,
            )
        )
        # update active sequences
        next_token_active = (
            (next_tokens != termination_token_id).squeeze(-1)
            & (next_tokens != eos_token_id).squeeze(-1)
        )
        active_seqs = active_seqs * next_token_active

        # add sampled token to decoder input
        decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)

        # check if all sequences have terminated
        if torch.all(~active_seqs):
            break

    return decoder_input_ids, log_pf
