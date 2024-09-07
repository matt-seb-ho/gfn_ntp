from types import MethodType
import json
import pickle
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from proof_flow.src.constants import (
    GFN_POLICY_ADAPTER_NAME,
)
from proof_flow.src.gfn_tuning.reward import NTPReward
from proof_flow.src.gfn_tuning.replay_buffer import ReplayBuffer, BUFFER_ENTRY_KEYS
from proof_flow.src.gfn_tuning.lean_data_module import NTPDataModule
from proof_flow.src.gfn_tuning.ntp import NeuralTheoremProvingTask
from proof_flow.src.utils import set_up_padding, repo_root

TEST_TRAINING_STEP = False
TEST_REPLAY_STEP = True
# relative to this file (proof_flow/scripts/gfn_tuning/train.py)
CONFIG_DIR = "../../../configs/"


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def train(config: DictConfig):
    pl.seed_everything(config.seed, workers=True)

    model, tokenizer = get_model(config)
    reward = get_reward(config, model, tokenizer)
    reward_buffer = ReplayBuffer(
        buffer_size=config.task.reward.buffer_size,
        termination_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        sim_tolerance=config.task.reward.buffer_sim_tolerance,
    )
    data = NTPDataModule(
        data_path=config.task.data.path,
        train_size=config.task.data.train_size,
    )
    data.setup()

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
        use_replay_tree=config.task.training.use_replay_tree,
        model_inference_batch_size=config.task.model.inf_batch_size,
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

    print("FINISHED INIT")
    thm0 = data.train_data[0]
    
    if TEST_TRAINING_STEP:
        # run a training step
        task.training_step(thm0, 0)
        # check results
        # pickle reward_buffer._buffer
        with open(repo_root() / "outputs/train_step0_rb.pkl", "wb") as f:
            pickle.dump(reward_buffer._buffer, f)
        # json dump proofs sans tensors
        sans_tensors = []
        skip_keys = {"state_tactic_tokens"}
        for trajectory in reward_buffer._buffer[thm0.uid]["proofs"]:
            entry = {
                k: trajectory[i]
                for i, k in enumerate(BUFFER_ENTRY_KEYS)
                if k not in skip_keys
            }
            sans_tensors.append(entry)
        filename = repo_root() / "outputs/train_step0_trajectories.json"
        with open(filename, 'w') as f:
            json.dump(sans_tensors, f, indent=4)
        print(f"Saved proof buffer to {filename}")

    elif TEST_REPLAY_STEP:
        # first populate the buffer with prior trajectories
        with open(repo_root() / "outputs/train_step0_rb.pkl", "rb") as f:
            reward_buffer._buffer = pickle.load(f)

        # run a training step with forced replay
        task.training_step(thm0, 0, force_replay=True)
    


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

    lora_config = hydra.utils.instantiate(config.task.model.lora_config)
    print(lora_config)
    # Wrap using Lora
    model = get_peft_model(
        model, 
        lora_config,
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
        model=model,
        tokenizer=tokenizer,
        # temperature is set dynamically
        # temperature=config.task.reward.temperature, 
        verifier_batch_size=config.task.reward.verifier_batch_size,
        verifier_adapter_name=config.task.reward.reward_model_adapter_name,
    )
    return reward


if __name__ == "__main__":
    train()
