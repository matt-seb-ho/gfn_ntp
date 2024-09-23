from types import MethodType
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from proof_flow.scripts.gfn_tuning.train import (
    get_model,
    get_reward
)
from icecream import ic
from proof_flow.src.gfn_tuning.reward import NTPReward
from proof_flow.src.gfn_tuning.replay_buffer import ReplayBuffer
from proof_flow.src.gfn_tuning.lean_data_module import NTPDataModule
from proof_flow.src.gfn_tuning.ntp import NeuralTheoremProvingTask
from proof_flow.src.utils import set_up_padding

# relative to this file (proof_flow/scripts/gfn_tuning/train.py)
CONFIG_DIR = "../../../configs/"


def compare_tensors(t1, t2):
    # basic info
    ic(t1.dtype, t2.dtype, t1.shape, t2.shape)
    # max/min of each tensor
    ic(t1.max().item(), t1.min().item(), t2.max().item(), t2.min().item())
    abs_diff = torch.abs(t1 - t2)
    ic(abs_diff.max().item(), abs_diff.mean().item())


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def main(config: DictConfig):
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
        replay_batch_size=config.task.training.replay_batch_size,
    )

    # Fix a bug that arises when using 4-bit quantized models.
    # It's caused by different operations being on different devices,
    # so we'll just deactivate lightning's automatic device placement
    # and let huggingface handle the dynamic device placement
    if config.task.training.use_4bit:
        task.to = MethodType(lambda s, _: s, task)
        task.cuda = MethodType(lambda s: s, task)
    

    thm0 = data.train_data[0]
    # check forward pass
    with torch.no_grad():
        t_logpf, log_r, extracted_ts = task.forward(thm0, pf_temperature=1.0)
        
        replay_tlogpf, replay_log_r = task.replay_trajectories(
            extracted_ts,
            batch_size=config.task.model.inf_batch_size,
        )

    # we expect the t_logpfs to be the same, and the log_r to be the same
    # I'm not quite sure what quantization does to the logits dtype
    ic(t_logpf, replay_tlogpf)
    compare_tensors(t_logpf, replay_tlogpf)
    compare_tensors(log_r, replay_log_r)
    

if __name__ == "__main__":
    main()
