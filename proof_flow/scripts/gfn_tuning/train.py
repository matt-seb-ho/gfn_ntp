import hydra
import json
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig
from typing import Optional
from peft import (
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
    get_peft_model, 
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from types import MethodType
from proof_flow.src.constants import (
    GFN_POLICY_ADAPTER_NAME,
)
from proof_flow.src.prompts import PROMPT_DICT
from proof_flow.src.gfn_tuning.reward import NTPReward
from proof_flow.src.gfn_tuning.replay_buffer import (
    BufferEntry,
    ReplayBuffer,
    extract_ground_truth_trajectory,
)
from proof_flow.src.gfn_tuning.lean_data_module import NTPDataModule
from proof_flow.src.gfn_tuning.ntp import NeuralTheoremProvingTask
from proof_flow.src.search.common import _HuggingFaceLM
from proof_flow.src.utils import (
    disable_tokenizer_parallelism,
    repo_root,
    set_up_padding,
    set_up_debug_logging,
)


# relative to this file (proof_flow/scripts/gfn_tuning/train.py)
CONFIG_DIR = "../../../configs/"


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def train(config: DictConfig):
    task, data, trainer = train_setup(config)
    trainer.fit(model=task, datamodule=data)


def train_setup(
    config: DictConfig
) -> tuple[NeuralTheoremProvingTask, NTPDataModule, pl.Trainer]:
    # misc setup
    pl.seed_everything(config.seed, workers=True)
    disable_tokenizer_parallelism()
    debug_log_level = set_up_debug_logging(config.task.debug_logger)

    # load data
    data = NTPDataModule(
        data_path=config.task.data.path,
        train_size=config.task.data.train_size,
        train_data_path=config.task.data.train_data_path,
        val_data_path=config.task.data.val_data_path,
        repeat_theorem_n_times=config.task.data.repeat_train_theorems,
    )
    data.setup("fit")
    val_probes = get_val_probes(config)
            
    # set up model, reward, and replay buffer
    model, tokenizer = get_model(config)
    reward = get_reward(config, model, tokenizer)
    reward_buffer = ReplayBuffer(
        buffer_size=config.task.reward.buffer_size,
        sim_tolerance=config.task.reward.buffer_sim_tolerance,
        tokenizer=tokenizer,
    )
    # optionally add seed trajectories to replay buffer
    if config.task.reward.buffer_seed_trajectory_file is not None:
        seed_file = repo_root() / config.task.reward.buffer_seed_trajectory_file
        with open(seed_file) as f:
            seed_trajectories = json.load(f)
        # expect seed_trajectories to be dict[str, list[list]]
        # where the innermost list is really a BufferEntry tuple
        # (json serializes BufferEntry named tuple as a list)
        for thm_uid, trajectories in seed_trajectories.items():
            trajectory_batch = [BufferEntry(*t) for t in trajectories]
            reward_buffer.add_batch(thm_uid, trajectory_batch)
    # optionally load gold trajectories (inserted into batch )
    gtt = get_ground_truth_trajectories(config)
    if config.task.gtt.seed_replay_buffer and gtt is not None:
        for tuid, gtt in gtt.items():
            reward_buffer.add_batch(tuid, [gtt])

    # set up task (LightningModule)
    tac_gen_prompt_template = PROMPT_DICT[config.task.prompts.tac_gen]
    search_params = hydra.utils.instantiate(
        config.task.search_eval.search_params
    )
    task = NeuralTheoremProvingTask(
        model=model,
        tokenizer=tokenizer,
        reward=reward,
        reward_buffer=reward_buffer,
        n_samples=config.task.training.n_samples,
        lr=config.task.training.lr,
        pf_temp_high=config.task.training.pf_temp_high,
        pf_temp_low=config.task.training.pf_temp_low,
        pf_temp_prob=config.task.training.pf_temp_prob,
        use_buffer_prob=config.task.training.use_buffer_prob,
        reward_temp_start=config.task.reward.temp_start,
        reward_temp_end=config.task.reward.temp_end,
        reward_temp_horizon=config.task.reward.temp_horizon,
        use_4bit=config.task.training.use_4bit,
        max_tactics=config.task.constraints.max_tactics,
        min_tactic_tokens=config.task.constraints.min_tactic_tokens,
        max_tactic_tokens=config.task.constraints.max_tactic_tokens,
        replay_batch_size=config.task.training.replay_batch_size,
        dojo_timeout=config.task.training.dojo_timeout,
        max_input_length=config.task.constraints.max_input_length,
        branch_only_at_root=config.task.training.branch_only_at_root,
        search_eval_probes=val_probes,
        search_eval_params=search_params,
        ckpt_dest=config.task.training.ckpt_dest,
        save_ckpt_on_val=config.task.training.save_ckpt_on_val,
        sanity_check_probes=config.task.search_eval.sanity_check_probe_count,
        debug_log_level=debug_log_level,
        tac_gen_prompt_template=tac_gen_prompt_template,
        ground_truth_trajectories=gtt,
        repeats_per_accumulated_batch=config.task.data.repeat_train_theorems,
        seq2seq=config.task.model.seq2seq,
        truncate_state=config.task.training.truncate_state,
    )

    # set up trainer
    trainer_logger = (
        config.logger 
        if isinstance(config.logger, bool) 
        else hydra.utils.instantiate(config.logger)
    )
    san_steps = config.task.training.num_sanity_val_steps
    trainer = pl.Trainer(
        accelerator=config.device.accelerator,
        max_epochs=config.task.training.epochs,
        accumulate_grad_batches=config.task.training.accumulate_grad_batches,
        logger=trainer_logger,
        callbacks=[hydra.utils.instantiate(c) for c in config.task.callbacks],
        val_check_interval=config.task.training.val_check_interval,
        check_val_every_n_epoch=config.task.training.check_val_every_n_epoch,
        gradient_clip_val=config.task.training.gradient_clip_val,
        num_sanity_val_steps=san_steps,
        log_every_n_steps=config.task.training.log_every_n_steps,
    )

    # Fix a bug that arises when using 4-bit quantized models.
    # It's caused by different operations being on different devices,
    # so we'll just deactivate lightning's automatic device placement
    # and let huggingface handle the dynamic device placement
    if config.task.training.use_4bit:
        task.to = MethodType(lambda s, _: s, task)
        task.cuda = MethodType(lambda s: s, task)

    return task, data, trainer


def get_model(config: DictConfig) -> tuple[_HuggingFaceLM, AutoTokenizer]:
    """
    loads the model and tokenizer and do some setup work
    - initialize bnb config
    - set up padding (add pad token, set side)
    - prepare for k-bit training
    - add policy adapters (if necessary)
    - remove dropout (from original code, not sure if needed)
    returns (model, tokenizer, reward_model, reward_tokenizer)
    """
    # use 4-bit quantization for lower memory use
    if config.task.training.use_4bit:
        # amortized paper uses bnb_4bit_compute_dtype="float16", not "bfloat16"
        bnb_config = hydra.utils.instantiate(config.task.model.bnb)
    else:
        bnb_config = None

    # get model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.task.model.hf_id, 
        # from amortized paper code, not sure if needed
        # add_bos_token=False,
    )
    if config.task.model.seq2seq:
        auto_model_cls = AutoModelForSeq2SeqLM
        peft_model_cls = PeftModelForSeq2SeqLM
    else:
        auto_model_cls = AutoModelForCausalLM
        peft_model_cls = PeftModelForCausalLM
    model = auto_model_cls.from_pretrained(
        config.task.model.hf_id, 
        torch_dtype="auto", # defer to torch_dtype from model config.json
        device_map="auto", 
        quantization_config=bnb_config,
    )
    # padding is needed for batch processing (e.g. reward computation)
    # llemma and deepseek models don't have padding tokens by default
    pad_side = "right" if config.task.model.seq2seq else "left"
    set_up_padding(model, tokenizer, padding_side=pad_side)

    # prepare model for k-bit training
    if config.task.training.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,  # Doesn't save memory when generating autoregressively compared to caching
        )

    # wrap with lora
    if config.task.model.use_lora:
        if config.task.model.initial_policy_adapter is None:
            # initialize a new adapter
            model = get_peft_model(
                model, 
                hydra.utils.instantiate(config.task.model.lora_config),
                adapter_name=GFN_POLICY_ADAPTER_NAME,
            )
        else:
            # load the specified adapter
            model = peft_model_cls.from_pretrained(
                model=model,
                model_id=config.task.model.initial_policy_adapter,
                adapter_name=GFN_POLICY_ADAPTER_NAME,
            )

    # Remove dropout
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0

    return model, tokenizer


def get_reward(
    config: DictConfig,
    model: _HuggingFaceLM,
    tokenizer: AutoTokenizer
) -> NTPReward:
    rm_cfg = config.task.reward.model
    rm_setup_options = {None, "base", "adapter", "independent"}
    assert rm_cfg.setup in rm_setup_options, (
        f"cfg.task.reward.model.setup must be one of {rm_setup_options}"
    )
    reward_model = None
    reward_tokenizer = tokenizer
    adapter_name = None
    reward_uses_seq2seq = config.task.model.seq2seq
    if rm_cfg.setup == "base":
        reward_model = model
    elif rm_cfg.setup == "adapter":
        # load in reward adapter
        adapter_name = rm_cfg.adapter.name
        model.load_adapter(
            rm_cfg.adapter.hf_id,
            adapter_name=adapter_name,
        )
        reward_model = model
    elif rm_cfg.setup == "independent":
        auto_cls = _get_auto_cls(rm_cfg.seq2seq, rm_cfg.peft)
        reward_model = auto_cls.from_pretrained(
            rm_cfg.hf_id,
            device_map="auto",
            torch_dtype="auto",
        )
        reward_tokenizer = (
            tokenizer if rm_cfg.share_tokenizer else
            AutoTokenizer.from_pretrained(rm_cfg.hf_id)
        )
        reward_uses_seq2seq = rm_cfg.seq2seq
    
    # if we are fully fine-tuning the policy with partial reward (non-binary)
    if (not config.task.model.use_lora) and rm_cfg.setup is not None:
        # there must be an independent reward model
        assert rm_cfg.hf_id is not None
        
    reward = NTPReward(
        setup=rm_cfg.setup,
        model=reward_model,
        tokenizer=reward_tokenizer,
        batch_size=config.task.reward.verifier_batch_size,
        adapter_name=rm_cfg.adapter.name,
        seq2seq=reward_uses_seq2seq,
        prompts_for_model=config.task.prompts.reward,
        use_sts_format=config.task.reward.use_sts_format,
        max_input_length=config.task.constraints.max_input_length,
        max_tactic_length=config.task.constraints.max_tactic_tokens,
        error_length_penalty_alpha=config.task.reward.error_length_penalty_a,
    )
    return reward


def get_val_probes(cfg: DictConfig):
    with open(repo_root() / cfg.task.search_eval.probe_file) as f:
        probes = json.load(f)
    # convert to list from {idx: thm} dict
    probes = list(probes.values())
    # limit number of probes
    if cfg.task.search_eval.probe_count is not None:
        probes = probes[:cfg.task.search_eval.probe_count]
    return probes


def get_ground_truth_trajectories(cfg: DictConfig) -> Optional[dict]:
    if cfg.task.gtt.file_path is None:
        return None
    
    gtt_file_path = repo_root() / cfg.task.gtt.file_path
    if cfg.task.gtt.write_to_file:
        trajectories = {}
        if cfg.task.data.path:
            _add_gtt_from_file(cfg.task.data.path, trajectories)
        else:
            assert cfg.task.data.train_data_path is not None
            assert cfg.task.data.val_data_path is not None
            _add_gtt_from_file(cfg.task.data.train_data_path, trajectories)
            _add_gtt_from_file(cfg.task.data.val_data_path, trajectories)
        with open(gtt_file_path, "w") as f:
            json.dump(trajectories, f, indent=2)
    else:
        with open(gtt_file_path) as f:
            json_trajectories = json.load(f)
        # convert to BufferEntry
        trajectories = {}
        for thm_uid, gtt in json_trajectories.items():
            trajectories[thm_uid] = BufferEntry(*gtt)
    return trajectories


def _add_gtt_from_file(file_path: str, trajectories: dict):
    with open(file_path) as f:
        thm_dicts = json.load(f)
    for thm_dict in thm_dicts.values():
        tuid, gtt = extract_ground_truth_trajectory(thm_dict)
        trajectories[tuid] = gtt


def _get_auto_cls(seq2seq: bool, peft: bool):
    if seq2seq:
        return PeftModelForSeq2SeqLM if peft else AutoModelForSeq2SeqLM
    else:
        return PeftModelForCausalLM if peft else AutoModelForCausalLM


if __name__ == "__main__":
    train()
