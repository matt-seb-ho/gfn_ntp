from types import MethodType

import hydra
from proof_flow.src.utils import set_up_padding
import pytorch_lightning as pl
import torch
# from lightning_module import NextSentenceGFNTask
from omegaconf import DictConfig, OmegaConf
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from proof_flow.constants import (
    GFN_POLICY_ADAPTER_NAME,
    REWARD_ADAPTER_NAME,
)
from proof_flow.src.gfn_tuning.reward import NTPReward
from proof_flow.src.gfn_tuning.replay_buffer import ReplayBuffer
from proof_flow.src.gfn_tuning.lean_data_module import NTPDataModule
from proof_flow.src.gfn_tuning.ntp import NeuralTheoremProvingTask

from .ppo import NTP_PPO

@hydra.main(version_base=None, config_path="/home/vincentzhu/gfn_ntp/configs/", config_name="example_train")
# @hydra.main(version_base=None, config_path="/Users/vincentwork/Documents/GFN_NTP/gfn_ntp/configs/", config_name="example_train")
def train(config: DictConfig):
    pl.seed_everything(config.seed, workers=True)

    model, tokenizer = get_model(config)
    reward = get_reward(config, model, tokenizer)
    reward_buffer = ReplayBuffer(
        buffer_size=config.task.replay_buffer.buffer_size,
        termination_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        sim_tolerance=config.task.replay_buffer.sim_tolerance,
    )
    data = NTPDataModule(
        data_path=config.task.data.path,
        tokenizer=tokenizer,
        train_size=config.task.data.train_size,
        # limit_theorems=config.task.data.limit_theorems,
    )

    data.setup("fit")
    train_probes = [data.train_data[i][0] for i in range(config.task.eval.n_probes)]
    val_probes = [data.val_data[i][0] for i in range(config.task.eval.n_probes)]

    task = NeuralTheoremProvingTask(
        model=model,
        tokenizer=tokenizer,
        reward=reward,
        reward_buffer=reward_buffer,
        n_samples=config.task.training.n_samples,
        lr=config.task.training.lr,
        subtb_lambda=config.task.training.subtb_lambda,
        pf_temp_high=config.task.training.pf_temp_high,
        pf_temp_low=config.task.training.pf_temp_low,
        pf_temp_prob=config.task.training.pf_temp_prob,
        use_buffer_prob=config.task.training.use_buffer_prob,
        reward_temp_start=config.task.reward.temp_start,
        reward_temp_end=config.task.reward.temp_end,
        reward_temp_horizon=config.task.reward.temp_horizon,
        train_probes=train_probes,
        val_probes=val_probes,
        use_4bit=config.task.training.use_4bit,
        max_tactics=config.task.max_tactics,
        min_tactic_tokens=config.task.min_tactic_tokens,
        max_tactic_tokens=config.task.max_tactic_tokens,
        use_hf_generate=config.task.use_hf_generate,
    )
    if config.task.name == "ntp":
        task = NTP_PPO(
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
            train_probes=train_probes,
            val_probes=val_probes,
            save_dir=config.task.training.save_dir,
            wandb_log=config.task.training.wandb_log,
            wandb_entity=config.task.training.wandb_entity,
            wandb_project=config.task.training.wandb_project,
        )

    trainer = pl.Trainer(
        accelerator=config.device.accelerator,
        max_epochs=config.task.training.epochs,
        accumulate_grad_batches=config.task.training.accumulate_grad_batches,
        logger=config.logger
        if isinstance(config.logger, bool)
        else hydra.utils.instantiate(config.logger),
        callbacks=[hydra.utils.instantiate(c) for c in config.task.callbacks],
    )

    # Fix a bug that arises when using 4-bit quantized models.
    # It's caused by different operations being on different devices,
    # so we'll just deactivate lightning's automatic device placement
    # and let huggingface handle the dynamic device placement
    if config.task.training.use_4bit:
        task.to = MethodType(lambda s, _: s, task)
        task.cuda = MethodType(lambda s: s, task)

    trainer.fit(model=task, datamodule=data)


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
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Get the model
    tokenizer = AutoTokenizer.from_pretrained(
        config.task.model.name, 
        # from original code, not sure if needed
        # add_bos_token=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.task.model.name, 
        device_map="auto", 
        quantization_config=bnb_config
    )

    # padding is needed for batch processing (e.g. reward computation)
    # llemma and deepseek models don't have padding tokens by default
    set_up_padding(model, tokenizer)

    # Prepare model for k-bit training
    if config.task.training.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,  # Doesn't save memory when generating autoregressively compared to caching
        )

    # Wrap using Lora
    model = get_peft_model(
        model, 
        hydra.utils.instantiate(config.task.model.lora_config),
        adapter_name=GFN_POLICY_ADAPTER_NAME,
    )
    
    # Load in reward adapter
    model.load_adapter(
        config.task.reward.adapter_name,
        adapter_name=REWARD_ADAPTER_NAME,
    )

    # Remove dropout
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0

    return model, tokenizer


def get_reward(config: DictConfig, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    model_loading_kwargs_dict = OmegaConf.to_container(
        config.task.reward.model_loading_kwargs,
        resolve=True,
    )
    reward = NTPReward(
        model=model,
        tokenizer=tokenizer,
        temperature=config.task.reward.temperature,
        verifier_batch_size=config.task.reward.verifier_batch_size,
        model_loading_kwargs=model_loading_kwargs_dict,
        verifier_adapter_name=REWARD_ADAPTER_NAME,
    )
    return reward


if __name__ == "__main__":
    train()
