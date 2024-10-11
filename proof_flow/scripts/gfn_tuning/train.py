import hydra
import json
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
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
)


# relative to this file (proof_flow/scripts/gfn_tuning/train.py)
CONFIG_DIR = "../../../configs/"


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def train(config: DictConfig):
    task, data, trainer = train_setup(config)
    trainer.fit(model=task, datamodule=data)


def train_setup(
    config: DictConfig
) -> tuple[NeuralTheoremProvingTask, NTPDataModule, pl.Trainer]:
    # misc setup
    pl.seed_everything(config.seed, workers=True)
    disable_tokenizer_parallelism()

    # load data
    data = instantiate(config.data)
    data.setup("fit")
    gtt = get_ground_truth_trajectories(config.ntp.gtt, config.data)
    seed_trajectories = gtt if config.ntp.gtt.seed_replay_buffer else None
            
    # set up model, reward, and replay buffer
    model, tokenizer = set_up_model_and_tokenizer(config.model)
    reward = set_up_reward(config, model, tokenizer)
    reward_buffer = ReplayBuffer(
        buffer_size=config.reward.buffer_size,
        sim_tolerance=config.reward.buffer_sim_tolerance,
        tokenizer=tokenizer,
        seed_file=config.reward.buffer_seed_trajectory_file,
        seed_trajectories=seed_trajectories,
    )

    # set up task (LightningModule)
    ntp_model_config = instantiate(config.ntp.ntp_config)
    search_params = instantiate(config.ntp.search_eval)
    task = NeuralTheoremProvingTask(
        model=model,
        tokenizer=tokenizer,
        reward=reward,
        reward_buffer=reward_buffer,
        config=ntp_model_config,
        ground_truth_trajectories=gtt,
        search_eval_params=search_params,
    )

    # set up trainer
    trainer_callbacks = [instantiate(c) for c in config.trainer.callbacks]
    trainer = instantiate(config.trainer, callbacks=trainer_callbacks)

    # Fix a bug that arises when using 4-bit quantized models.
    # It's caused by different operations being on different devices,
    # so we'll just deactivate lightning's automatic device placement
    # and let huggingface handle the dynamic device placement
    if config.model.use_4bit:
        task.to = MethodType(lambda s, _: s, task)
        task.cuda = MethodType(lambda s: s, task)

    return task, data, trainer


def set_up_model_and_tokenizer(
    config: DictConfig
) -> tuple[_HuggingFaceLM, AutoTokenizer]:
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
    if config.use_4bit:
        # amortized paper uses bnb_4bit_compute_dtype="float16", not "bfloat16"
        bnb_config = instantiate(config.bnb)
    else:
        bnb_config = None

    # get model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id, 
        # from amortized paper code, not sure if needed
        # add_bos_token=False,
    )
    if config.seq2seq:
        auto_model_cls = AutoModelForSeq2SeqLM
        peft_model_cls = PeftModelForSeq2SeqLM
    else:
        auto_model_cls = AutoModelForCausalLM
        peft_model_cls = PeftModelForCausalLM
    model = auto_model_cls.from_pretrained(
        config.hf_id, 
        torch_dtype="auto", # defer to torch_dtype from model config.json
        device_map="auto", 
        quantization_config=bnb_config,
    )

    # padding is needed for batch processing (e.g. reward computation)
    # llemma and deepseek models don't have padding tokens by default
    pad_side = "right" if config.seq2seq else "left"
    set_up_padding(model, tokenizer, padding_side=pad_side)

    # prepare model for k-bit training
    if config.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,  
            # gradient checkpointing doesn't save memory 
            # when generating autoregressively (compared to caching)
        )

    # wrap with lora
    if config.use_lora:
        if config.initial_policy_adapter is None:
            # initialize a new adapter
            model = get_peft_model(
                model, 
                instantiate(config.lora_config),
                adapter_name=GFN_POLICY_ADAPTER_NAME,
            )
        else:
            # load the specified adapter
            model = peft_model_cls.from_pretrained(
                model=model,
                model_id=config.initial_policy_adapter,
                adapter_name=GFN_POLICY_ADAPTER_NAME,
            )

    # remove dropout
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0

    return model, tokenizer


def set_up_reward(
    config: DictConfig,
    model: _HuggingFaceLM,
    tokenizer: AutoTokenizer
) -> NTPReward:
    rm_cfg = config.reward.model
    rm_setup_options = {None, "base", "adapter", "independent"}
    assert rm_cfg.setup in rm_setup_options, (
        f"cfg.reward.model.setup must be one of {rm_setup_options}"
    )
    reward_model = None
    reward_tokenizer = tokenizer
    adapter_name = None
    reward_uses_seq2seq = config.model.seq2seq
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
    if (not config.model.use_lora) and rm_cfg.setup is not None:
        # there must be an independent reward model
        assert rm_cfg.hf_id is not None
        
    reward = NTPReward(
        setup=rm_cfg.setup,
        model=reward_model,
        tokenizer=reward_tokenizer,
        batch_size=config.reward.verifier_batch_size,
        adapter_name=rm_cfg.adapter.name,
        seq2seq=reward_uses_seq2seq,
        prompts_for_model=config.reward.prompts_for_model,
        use_sts_format=config.reward.use_sts_format,
        max_input_length=config.ntp.ntp_config.max_input_length,
        max_tactic_length=config.ntp.ntp_config.max_tactic_tokens,
        error_length_penalty_alpha=config.reward.error_length_penalty_a,
    )
    return reward


def get_ground_truth_trajectories(
    gtt_cfg: DictConfig,
    data_cfg: DictConfig,
) -> Optional[dict]:
    if gtt_cfg.file_path is None:
        return None
    
    gtt_file_path = repo_root() / gtt_cfg.file_path
    if gtt_cfg.write_to_file:
        trajectories = {}
        if data_cfg.data_path:
            _add_gtt_from_file(data_cfg.data_path, trajectories)
        else:
            assert data_cfg.train_data_path is not None
            assert data_cfg.val_data_path is not None
            _add_gtt_from_file(data_cfg.train_data_path, trajectories)
            _add_gtt_from_file(data_cfg.val_data_path, trajectories)
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
