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
    ReplayBuffer,
    extract_ground_truth_trajectory,
)
from proof_flow.src.gfn_tuning.lean_data_module import NTPDataModule
from proof_flow.src.gfn_tuning.ntp import NeuralTheoremProvingTask
from proof_flow.src.utils import (
    disable_tokenizer_parallelism,
    repo_root,
    set_up_padding,
    set_up_debug_logging,
)


# relative to this file (proof_flow/scripts/gfn_tuning/train.py)
CONFIG_DIR = "../../../configs/"


def get_model(config: DictConfig):
    """
    loads the model and tokenizer and do some setup work
    - initialize bnb config
    - set up padding (add pad token, set side)
    - prepare for k-bit training
    - add policy adapters
    - load (but not set as active) reward adapter
    - remove dropout (from original code, not sure if needed)
    """
    
    # Use 4-bit quantization for lower memory use
    if config.task.training.use_4bit:
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype="float16",
        #     bnb_4bit_use_double_quant=True,
        # )
        # config has all the same options as above
        # EXCEPT bnb_4bit_compute_dtype is "bfloat16" instead of "float16"
        bnb_config = hydra.utils.instantiate(config.task.model.bnb)
    else:
        bnb_config = None

    # Get the model
    tokenizer = AutoTokenizer.from_pretrained(
        config.task.model.name, 
        # from original code, not sure if needed
        # add_bos_token=False,
    )
    
    if config.task.model.seq2seq:
        auto_model_cls = AutoModelForSeq2SeqLM
        peft_model_cls = PeftModelForSeq2SeqLM
    else:
        auto_model_cls = AutoModelForCausalLM
        peft_model_cls = PeftModelForCausalLM
        
    model = auto_model_cls.from_pretrained(
        config.task.model.name, 
        torch_dtype="auto", # defer to torch_dtype from model config.json
        device_map="auto", 
        quantization_config=bnb_config,
    )

    # padding is needed for batch processing (e.g. reward computation)
    # llemma and deepseek models don't have padding tokens by default
    pad_side = "right" if config.task.model.seq2seq else "left"
    set_up_padding(model, tokenizer, padding_side=pad_side)

    # Prepare model for k-bit training
    if config.task.training.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,  # Doesn't save memory when generating autoregressively compared to caching
        )

    # Wrap using Lora
    if config.task.model.initialize_policy_adapter_from_pretrained is None:
        # if no initialization is specified, create a new adapter from config
        model = get_peft_model(
            model, 
            hydra.utils.instantiate(config.task.model.lora_config),
            adapter_name=GFN_POLICY_ADAPTER_NAME,
        )
    else:
        # otherwise, load the specified adapter
        model = peft_model_cls.from_pretrained(
            model=model,
            model_id=config.task.model.initialize_policy_adapter_from_pretrained,
            adapter_name=GFN_POLICY_ADAPTER_NAME,
        )
    
    # Load in reward adapter
    if config.task.reward.reward_model_hf_id is not None:
        model.load_adapter(
            config.task.reward.reward_model_hf_id,
            adapter_name=config.task.reward.reward_model_adapter_name,
        )

    # Remove dropout
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0

    return model, tokenizer


def get_reward(config: DictConfig, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    reward = NTPReward(
        model,
        tokenizer,
        # temperature is set dynamically
        # temperature=config.task.reward.temperature, 
        verifier_batch_size=config.task.reward.verifier_batch_size,
        verifier_adapter_name=config.task.reward.reward_model_adapter_name,
        seq2seq=config.task.model.seq2seq,
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
        with open(cfg.task.data.path or cfg.task.data.train_data_path) as f:
            thm_dicts = json.load(f)
        trajectories = {}
        for thm_dict in thm_dicts.values():
            tuid, gtt = extract_ground_truth_trajectory(thm_dict)
            trajectories[tuid] = gtt
        with open(gtt_file_path, "w") as f:
            json.dump(trajectories, f, indent=2)
    else:
        with open(gtt_file_path) as f:
            trajectories = json.load(f)
    return trajectories


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
        repeat_theorem_n_times=config.task.training.accumulate_grad_batches,
    )
    data.setup("fit")
    val_probes = get_val_probes(config)
            
    # set up model, reward, and replay buffer
    model, tokenizer = get_model(config)
    reward = get_reward(config, model, tokenizer)
    reward_buffer = ReplayBuffer(
        buffer_size=config.task.reward.buffer_size,
        termination_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        sim_tolerance=config.task.reward.buffer_sim_tolerance,
        tokenizer=tokenizer,
    )
    tac_gen_prompt_template = PROMPT_DICT[config.task.prompts.tac_gen]
    # - optionally load ground truth trajectories
    ground_truth_trajectories = get_ground_truth_trajectories(
        config,
        tokenizer,
        tac_gen_prompt_template,
    )
    # - optionally load seed trajectories
    if config.task.reward.buffer_seed_trajectory_file is not None:
        with open(repo_root() / config.task.reward.buffer_seed_trajectory_file) as f:
            seed_trajectories = json.load(f)
        # to make the trajectories serializable,
        # we converted state_tactic_tokens tensor -> list.
        # we should convert them back to tensor here
        for thm_trajectories in seed_trajectories.values():
            for t in thm_trajectories:
                t["state_tactic_tokens"] = [
                    torch.tensor(stt) for stt in t["state_tactic_tokens"]
                ]
        for thm_uid, trajectories in seed_trajectories.items():
            reward_buffer.add_batch(thm_uid, trajectories)

    # set up task
    search_params = hydra.utils.instantiate(config.task.search_eval.search_params)
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
        model_inference_batch_size=config.task.model.inf_batch_size,
        branch_only_at_root=config.task.training.branch_only_at_root,
        dojo_timeout=config.task.training.dojo_timeout,
        search_eval_probes=val_probes,
        ckpt_dest=config.task.training.ckpt_dest,
        save_ckpt_on_val=config.task.training.save_ckpt_on_val,
        sanity_check_probes=config.task.search_eval.sanity_check_probe_count,
        debug_log_level=debug_log_level,
        tac_gen_prompt_template=tac_gen_prompt_template,
        search_eval_params=search_params,
        ground_truth_trajectories=ground_truth_trajectories,
        accumulate_grad_batches=config.task.training.accumulate_grad_batches,
        use_log_z_cache=config.task.training.use_log_z_cache,
        seq2seq=config.task.model.seq2seq,
    )

    # set up trainer
    trainer_logger = (
        config.logger 
        if isinstance(config.logger, bool) 
        else hydra.utils.instantiate(config.logger)
    )
    # san_steps = config.task.training.num_sanity_val_steps or 2
    trainer = pl.Trainer(
        accelerator=config.device.accelerator,
        max_epochs=config.task.training.epochs,
        accumulate_grad_batches=config.task.training.accumulate_grad_batches,
        logger=trainer_logger,
        callbacks=[hydra.utils.instantiate(c) for c in config.task.callbacks],
        val_check_interval=config.task.training.val_check_interval,
        check_val_every_n_epoch=config.task.training.check_val_every_n_epoch,
        gradient_clip_val=config.task.training.gradient_clip_val,
        # num_sanity_val_steps=san_steps,
        num_sanity_val_steps=0,
    )

    # Fix a bug that arises when using 4-bit quantized models.
    # It's caused by different operations being on different devices,
    # so we'll just deactivate lightning's automatic device placement
    # and let huggingface handle the dynamic device placement
    if config.task.training.use_4bit:
        task.to = MethodType(lambda s, _: s, task)
        task.cuda = MethodType(lambda s: s, task)

    return task, data, trainer
    
    

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def train(config: DictConfig):
    task, data, trainer = train_setup(config)
    trainer.fit(model=task, datamodule=data)


if __name__ == "__main__":
    train()
