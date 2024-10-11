import argparse
import json
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from icecream import ic
import bitsandbytes as bnb
from proof_flow.src.constants import TACTIC_DELIMITER
from proof_flow.src.gfn_tuning.replay_buffer import ReplayBuffer, BufferEntry
from proof_flow.src.utils import repo_root, get_config
from proof_flow.scripts.gfn_tuning.train import train_setup

# relative to this file (proof_flow/scripts/gfn_tuning/train.py)
CONFIG_DIR = "../../../configs/"
N_SAMPLES_OVERRIDE = 6
INF_BATCH_SIZE_OVERRIDE = 6


def multiply_state(state, n):
    return "\n".join([state] * n)

device = torch.device("cuda")
def gb_allocated():
    return torch.cuda.memory_allocated(device) / 1e9

def main(config: DictConfig, n_samples_override, inf_batch_size_override):
    task, data, trainer = train_setup(config)
    model = task.model
    tokenizer = task.tokenizer

    model_mem = gb_allocated()

    # first construct the longest possible trajectory
    # with open(repo_root() / "data/longest_input_data.json") as f:
    with open(repo_root() / "data/max_batch_reprover_history.json") as f:
        input_data = json.load(f)
    # text = input_data["ll_prompt"] + input_data["tactic"]

    max_state = input_data["state_extended"]
    max_tactic = input_data["tactic"]
    
    proof = TACTIC_DELIMITER.join([max_tactic] * 3)
    trajectory = BufferEntry(
        log_r=0,
        proof=proof,
        states=[max_state] * 4,
    )
    task.reward_buffer.add_batch("longest_input", [trajectory])
    task.cfg.use_buffer_prob = 1
    task.max_batch_testing = n_samples_override
    task.cfg.replay_batch_size = inf_batch_size_override
    task.ground_truth_trajectories["longest_input"] = trajectory
    
    def thm0():
        pass
    thm0.uid = "longest_input"
    print(len(task.reward_buffer._buffer[thm0.uid]))

    # run a training step with forced replay
    # - do a backward step with optimizers to ensure we don't oom there either
    opt = bnb.optim.PagedAdamW8bit(model.parameters(), lr=task.cfg.lr)
    ic(opt)
    opt.zero_grad()
    loss = task.training_step([thm0], 0)

    total_mem = gb_allocated()
    tensor_mem = total_mem - model_mem
    print(f"total memory: {total_mem:.3f} GB, tensor memory: {tensor_mem:.3f} GB")
    

if __name__ == "__main__":
    # psr = argparse.ArgumentParser()
    # psr.add_argument("--n_samples", '-n', type=int, default=N_SAMPLES_OVERRIDE)
    # psr.add_argument("--inf_batch_size", '-b', type=int, default=INF_BATCH_SIZE_OVERRIDE)
    # args = psr.parse_args()
    
    config = get_config(config_name="train_five")
    main(
        config, 
        n_samples_override=N_SAMPLES_OVERRIDE,
        inf_batch_size_override=INF_BATCH_SIZE_OVERRIDE,
    )
