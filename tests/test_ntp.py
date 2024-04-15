import pytest
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model
import hydra
import torch

from src.gfn_tuning.ntp import NeuralTheoremProvingTask
from src.gfn_tuning.reward import NTPReward
from src.gfn_tuning.replay_buffer import ReplayBuffer

# time to use fixtures for the model itself
"""
Constructor signature for NTP:

    def __init__(
        self,
        model,
        tokenizer,
        reward,
        reward_buffer: ReplayBuffer,
        n_samples,
        lr,
        subtb_lambda,
        pf_temp_high,
        pf_temp_low,
        pf_temp_prob,
        use_buffer_prob,
        min_sentence_len,
        max_sentence_len,
        reward_temp_start,
        reward_temp_end,
        reward_temp_horizon,
        illegal_token_mask,
        train_probes=None,
        val_probes=None,
        use_4bit=False,
        max_steps=3,
    ):
"""

BASE_MODEL_ID = "EleutherAI/llemma_7b"
VERIFIER_ADAPTER_ID = "msho/llemma_dpo_sampled"

"""
@pytest.fixture(scope="session")
def ntp_task_module():
    # load config
    with hydra.initialize(version_base=None, config_path="../configs"):
        config = hydra.compose(config_name="config") 
    
    # initialize base model and tokenizer
    model = AutoModel.from_pretrained(BASE_MODEL_ID, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

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
    """

    return NeuralTheoremProvingTask()
