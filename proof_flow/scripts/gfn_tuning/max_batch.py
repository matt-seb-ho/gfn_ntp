import argparse
from types import MethodType
import json
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from icecream import ic
from proof_flow.src.gfn_tuning.replay_buffer import ReplayBuffer
from proof_flow.src.gfn_tuning.lean_data_module import NTPDataModule
from proof_flow.src.gfn_tuning.ntp import NeuralTheoremProvingTask
from proof_flow.src.utils import repo_root, get_config
from proof_flow.scripts.gfn_tuning.train import get_model, get_reward

# relative to this file (proof_flow/scripts/gfn_tuning/train.py)
CONFIG_DIR = "../../../configs/"
N_SAMPLES_OVERRIDE = 32
INF_BATCH_SIZE_OVERRIDE = 32


def main(config: DictConfig, n_samples_override, inf_batch_size_override):
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
        # n_samples=config.task.training.n_samples,
        n_samples=n_samples_override,
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
        # model_inference_batch_size=config.task.model.inf_batch_size,
        model_inference_batch_size=inf_batch_size_override,
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



    # first construct the longest possible trajectory
    with open(repo_root() / "data/longest_input_data.json") as f:
        input_data = json.load(f)
    text = input_data["prompt"] + input_data["tactic"]
    state_tactic_tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
    ic(state_tactic_tokens.shape)
    
    # trajectories.append({
    #     "theorem_id": theorem_id,
    #     "states": states.copy(),
    #     # root node has an empty string tactic, so we skip it
    #     "tactics": tactics[1:],
    #     "proof": TACTIC_DELIMITER.join(tactics[1:]),
    #     "state_tactic_tokens": parent_tactic_tokens[1:], # consider deepcopy...
    #     "prompt_lengths": prompt_lengths[1:],
    #     "log_r": node.log_r,
    # })
    tactics = [input_data["tactic"]] * 3
    prompt_length = len(tokenizer.encode(input_data["prompt"]))
    ic(prompt_length)
    trajectory = {
        "theorem_id": "longest_input",
        "states": [input_data["prompt"]] * 4,
        "tactics": tactics,
        "proof": "\n".join(tactics),
        "state_tactic_tokens": [state_tactic_tokens] * 3,
        "prompt_lengths": [prompt_length] * 3,
        "log_r": 0.0,
    }
    reward_buffer.add_batch("longest_input", [trajectory])
    
    def thm0():
        pass
    thm0.uid = "longest_input"

    # run a training step with forced replay
    # - do a backward step with optimizers to ensure we don't oom there either
    opt = task.optimizers()
    ic(opt)
    opt.zero_grad()
    loss = task.training_step(thm0, 0, force_replay=True)
    task.manual_backward(loss)
    opt.step()

    print(f"REPLAY PASSED WITH {n_samples_override} samples and {inf_batch_size_override} inference batch size")
    

if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument("--n_samples", '-n', type=int, default=N_SAMPLES_OVERRIDE)
    psr.add_argument("--inf_batch_size", '-b', type=int, default=INF_BATCH_SIZE_OVERRIDE)
    args = psr.parse_args()
    
    config = get_config(config_path=CONFIG_DIR, config_name="train")
    main(
        config, 
        n_samples_override=args.n_samples, 
        inf_batch_size_override=args.inf_batch_size
    )
