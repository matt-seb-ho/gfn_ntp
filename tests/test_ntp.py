import sys
# from loguru import logger
# logger.remove()
# logger.add(sys.stderr, level="DEBUG")

import pytest
from transformers import (
    AutoModel, 
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import get_peft_model
import hydra
import torch
import os
import json
from time import perf_counter

from src.gfn_tuning.lean_data_module import NTPDataModule
from src.gfn_tuning.ntp import NeuralTheoremProvingTask, lean_context
from src.gfn_tuning.reward import NTPReward
from src.gfn_tuning.replay_buffer import ReplayBuffer
from src.utils import make_path_relative_to_repo, load_github_access_token

load_github_access_token()
from lean_dojo import TacticState, LeanGitRepo, Theorem, is_available_in_cache # isort: skip

BASE_MODEL_ID = "EleutherAI/llemma_7b"
VERIFIER_ADAPTER_ID = "msho/llemma_dpo_sampled"

@pytest.fixture(scope="session")
def configs():
    # load config
    with hydra.initialize(version_base=None, config_path="../configs"):
        config = hydra.compose(config_name="train") 
    if config.task.training.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    return config, bnb_config

@pytest.fixture(scope="session")
def model_and_tokenizer(configs):
    # NOTE: Loads in model *without* adapters
    config, bnb_config = configs
    model_id = config.task.model.name
    start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id, 
        device_map="auto",
        quantization_config=bnb_config,
    )
    print(f"Model loading time: {perf_counter() - start}")
    return model, tokenizer

@pytest.fixture(scope="session")
def ntp_task_module(configs, model_and_tokenizer):
    config, _ = configs
    model, tokenizer = model_and_tokenizer

    # prep generator adapters
    model = get_peft_model(
        model=model,
        peft_config=hydra.utils.instantiate(config.task.model.lora_config),
    )
    print(f"Active adapter: {model.active_adapter}")
    
    # remove dropout (see original train.py)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    # end step generation at newline char
    end_of_step_token_id = tokenizer.encode("assumption\n", add_special_tokens=False)[-1]
    
    # initialize reward and replay buffer
    # - load trained verifier adapters
    model.load_adapter(VERIFIER_ADAPTER_ID)
    reward = NTPReward(
        model=model,
        tokenizer=tokenizer,
        verifier_adapter_name=VERIFIER_ADAPTER_ID,
    )
    replay_buffer = ReplayBuffer(
        buffer_size=10, 
        termination_token_id=end_of_step_token_id
    )
    n_samples = [3, 1, 1]

    # for the rest, we'll just use values from the original config values

    ntp_task = NeuralTheoremProvingTask(
        model=model,
        tokenizer=tokenizer,
        reward=reward,
        reward_buffer=replay_buffer,
        n_samples=n_samples,
        lr=config.task.training.lr,
        subtb_lambda=config.task.training.subtb_lambda,
        pf_temp_high=config.task.training.pf_temp_high,
        pf_temp_low=config.task.training.pf_temp_low,
        pf_temp_prob=config.task.training.pf_temp_prob,
        use_buffer_prob=config.task.training.use_buffer_prob,
        min_sentence_len=config.task.constraints.min_sentence_len,
        max_sentence_len=config.task.constraints.max_sentence_len,
        reward_temp_start=config.task.reward.temp_start,
        reward_temp_end=config.task.reward.temp_end,
        reward_temp_horizon=config.task.reward.temp_horizon,
        # illegal_token_mask=illegal_token_mask,
        # train_probes=train_probes,
        # val_probes=val_probes,
        illegal_token_mask=None,
        train_probes=None,
        val_probes=None,
        diversity_metric=config.task.eval.diversity_metric,
        use_4bit=config.task.training.use_4bit,
    )
    return ntp_task

def load_test_theorem(method="direct", traced_theorem_only=False):
    data_path = make_path_relative_to_repo("tests/test_data")
    with open(os.path.join(data_path, "test.json")) as f:
        # traced theorem
        tt = json.load(f)[0] 
    if traced_theorem_only:
        return None, tt
    if method == "direct":
        repo = LeanGitRepo(tt["url"], tt["commit"])
        thm = Theorem(repo=repo, file_path=tt["file_path"], full_name=tt["full_name"])
    elif method == "data module":
        data = NTPDataModule(data_path=data_path, train_size=1)
        data.setup("test")
        train_data = data.train_dataloader()
        thm = data.train_data[0]
    return thm, tt

def test_run_tactic():
    thm, traced_theorem = load_test_theorem("direct")
    traced_tactic = traced_theorem["traced_tactics"][0]
    tactic = traced_tactic["tactic"]
    
    with lean_context(thm, None) as (dojo, root):
        next_state = dojo.run_tac(root.state, tactic)
        print(
            f"Current state: {root.state.pp}\n"
            f"Running tactic: {tactic}\n"
            f"Next state: {getattr(next_state, 'pp', None)}"
        )
        assert isinstance(next_state, TacticState)
        assert next_state.pp == traced_tactic["state_after"]


def test_generate_step(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    _, traced_theorem = load_test_theorem(traced_theorem_only=True)
    traced_tactic = traced_theorem["traced_tactics"][0]
    goals = traced_tactic["state_before"]
    prompt = NeuralTheoremProvingTask.format_prompt(goals)
    encoded_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"]
    print(prompt)
    print(encoded_prompt.shape)
