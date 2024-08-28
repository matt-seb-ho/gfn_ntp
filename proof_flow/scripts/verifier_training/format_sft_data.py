import argparse
import json
from typing import Optional, Union

from datasets import Dataset, DatasetDict
from loguru import logger
from tqdm import tqdm

from proof_flow.src.utils import get_config, repo_root
from proof_flow.src.gfn_tuning.rm_prompts import (
    INSTRUCTION_PROMPT_TEMPLATE, 
    INSTRUCTION_COMPLETION_TEMPLATE_WITH_NEXT_STATE
)

MARK_START_SYMBOL = "<a>"
MARK_END_SYMBOL = "</a>"

# SFTTrainer instruction format 
# {"prompt": "<prompt text>", "completion": "<ideal generated text>"}

def remove_marks(s: str) -> str:
    """Remove all :code:`<a>` and :code:`</a>` from ``s``."""
    return s.replace(MARK_START_SYMBOL, "").replace(MARK_END_SYMBOL, "")


def preprocess_data(data_path: str) -> list[dict[str, str]]:
    data = []
    with open(data_path) as f:
        theorem_data = json.load(f)
    for thm_idx, thm in tqdm(theorem_data.items()):
        for tac_idx, tac in enumerate(thm["traced_tactics"]):
            tactic = remove_marks(tac["tactic"])
            data.append(
                {
                    "idx": thm_idx,
                    "tac_idx": tac_idx,
                    "url": thm["url"],
                    "commit": thm["commit"],
                    "file_path": thm["file_path"],
                    "full_name": thm["full_name"],
                    "state": tac["state_before"],
                    "tactic": tactic,
                    "next_state": tac["state_after"],
                }
            )
    logger.info(f"{len(data)} examples loaded")
    return data


def format_sft_dataset(
    records: list[dict[str, str]], 
    output_path: Optional[str] = None,
    train_size: Optional[Union[int, float]] = 0.8,
    include_next_state: bool = False,
    include_theorem_info: bool = False,
) -> DatasetDict:
    # format state, tactic [, next_state] as prompt, completion
    prompts = [INSTRUCTION_PROMPT_TEMPLATE.format(state=r["state"]) for r in records]
    if include_next_state:
        completions = [
            INSTRUCTION_COMPLETION_TEMPLATE_WITH_NEXT_STATE.format(
                tactic=r["tactic"], 
                next_state=r["next_state"]
            ) for r in records
        ]
    else:
        completions = [r["tactic"] for r in records]
    # convert to arrow dataset, create train-test split, and save to disk
    if include_theorem_info:
        dataset_dict = {
            "prompt": prompts,
            "completion": completions,
            "idx": [],
            "tac_idx": [],
            "url": [],
            "commit": [],
            "file_path": [],
        }
        for r in records:
            dataset_dict["idx"].append(r["idx"])
            dataset_dict["tac_idx"].append(r["tac_idx"])
            dataset_dict["url"].append(r["url"])
            dataset_dict["commit"].append(r["commit"])
            dataset_dict["file_path"].append(r["file_path"])
        dataset = Dataset.from_dict(dataset_dict)
    else:
        dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
    dataset = dataset.train_test_split(train_size=train_size, seed=42)
    if output_path is not None:
        dataset.save_to_disk(output_path)
    return dataset


if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument("--config", type=str, default="verifier_training")
    args = psr.parse_args()

    # read options from config file
    config = get_config(config_name=args.config)
    
    # load and preprocess data
    records = preprocess_data(repo_root() / config.sft.data.raw_data)

    # format and save dataset
    format_sft_dataset(
        records, 
        repo_root() / config.sft.data.formatted_dataset_dir, 
        train_size=config.sft.data.train_size, 
        include_next_state=config.sft.data.include_next_state,
    )
    logger.info("Data preparation complete")
