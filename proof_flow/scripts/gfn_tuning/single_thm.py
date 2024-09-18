import hydra
import json
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig
from peft import (
    PeftModelForCausalLM,
    get_peft_model, 
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from types import MethodType
from proof_flow.scripts.gfn_tuning.train import train_setup
from proof_flow.src.constants import (
    GFN_POLICY_ADAPTER_NAME,
)
from proof_flow.src.prompts import (
    DEEPSEEK_RM_ST_PROMPT_TEMPLATE_V2,
)
from proof_flow.src.gfn_tuning.reward import NTPReward
from proof_flow.src.gfn_tuning.replay_buffer import ReplayBuffer
from proof_flow.src.gfn_tuning.lean_data_module import NTPDataModule
from proof_flow.src.gfn_tuning.ntp import NeuralTheoremProvingTask


# relative to this file (proof_flow/scripts/gfn_tuning/train.py)
CONFIG_DIR = "../../../configs/"


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train1thm")
def train(config: DictConfig):
    task, data, trainer = train_setup(config)
    trainer.fit(model=task, datamodule=data)


if __name__ == "__main__":
    train()
