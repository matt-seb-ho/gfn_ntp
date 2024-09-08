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
from proof_flow.scripts.gfn_tuning.train import get_model, get_reward

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
        with open(repo_root() / "outputs/train_step0_rb_v2.pkl", "wb") as f:
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
        filename = repo_root() / "outputs/train_step0_trajectories_v2.json"
        with open(filename, 'w') as f:
            json.dump(sans_tensors, f, indent=4)
        print(f"Saved proof buffer to {filename}")

    elif TEST_REPLAY_STEP:
        # first populate the buffer with prior trajectories
        with open(repo_root() / "outputs/train_step0_rb.pkl", "rb") as f:
            reward_buffer._buffer = pickle.load(f)

        # run a training step with forced replay
        task.training_step(thm0, 0, force_replay=True)
    

if __name__ == "__main__":
    train()
