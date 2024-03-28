import json
from time import perf_counter
from typing import Optional
from datasets import Dataset, load_dataset
from tqdm import tqdm
from loguru import logger
from icecream import ic

from utils import prepend_repo_root, _pp
from prompts import INSTRUCTION_PROMPT_TEMPLATE
from lean_dojo_utils import (
    format_state,
    format_tactic,
    remove_marks,
)

# from lean_dojo_dataset import GeneratorDataset

# SFTTrainer instruction format 
# {"prompt": "<prompt text>", "completion": "<ideal generated text>"}

def preprocess_data(data_path: str, normalize_tactics: bool, keep_marks: bool):
    """
    data_path: str = pointing to json file containing theorems
    normalize_tactics: bool = whether to replace consecutive whitespace chars with single space
    keep_marks: bool = whether to wrap premises with <a></a> tags
    """
    data = []
    with open(data_path) as f:
        theorem_data = json.load(f)

    # for thm in tqdm(json.load(open(data_path))):
    for thm in tqdm(theorem_data):
        for tac in thm["traced_tactics"]:
            if "annotated_tactic" in tac:
                tactic = format_tactic(*tac["annotated_tactic"], normalize_tactics)
            else:
                tactic = format_tactic(tac["tactic"], [], normalize_tactics)
            if not keep_marks:
                tactic = remove_marks(tactic)
            data.append(
                {
                    "url": thm["url"],
                    "commit": thm["commit"],
                    "file_path": thm["file_path"],
                    "full_name": thm["full_name"],
                    "state": format_state(tac["state_before"]),
                    "tactic": tactic,
                }
            )

    logger.info(f"{len(data)} examples loaded")
    return data


def convert_to_instruction_pair(sample):
    return {
        "prompt": INSTRUCTION_PROMPT_TEMPLATE.format(state=sample["state"]),
        "completion": sample["tactic"]
    }

def prep_dataset(initial_ds: list[dict[str, str]], output_filename: Optional[str] = None):
    prompts = [INSTRUCTION_PROMPT_TEMPLATE.format(state=s["state"]) for s in initial_ds]
    completions = [s["tactic"] for s in initial_ds]
    ds = Dataset.from_dict({"prompt": prompts, "completion": completions})
    if output_filename is not None:
        ds.to_json(output_filename, orient="records")
    return ds 


def load_sft_data(path):
    dataset = load_dataset("json", data_files=path, split="train")
    # -- sanity check --
    # ic(len(dataset))
    # ic(dataset[0])
    return dataset


def sft_subset(dataset, size, seed=42, file=None):
    subset = dataset.train_test_split(test_size=size, seed=seed)["test"]
    if file:
        subset.to_json(file, orient="records")
    return subset


def main():
    RANDOM_TRAIN_DATA_PATH = prepend_repo_root("data/leandojo_benchmark_4/random/train.json")
    NOVELP_TRAIN_DATA_PATH = prepend_repo_root("data/leandojo_benchmark_4/novel_premises/train.json")
    # sfttif: SFTTrainer Instruction Format
    OUTPUT_PATH = prepend_repo_root("data/sfttif_random_train.json")
    thm_data = preprocess_data(RANDOM_TRAIN_DATA_PATH, True, False)
    
    # -- sanity check --
    # ic(len(thm_data))
    # ic(thm_data[0])
    # ic(thm_data[0]["tactic"])

    # instr_sample0 = convert_to_instruction_pair(thm_data[0])
    # print(instr_sample0["prompt"])
    # print(instr_sample0["completion"])
    
    train_dataset = prep_dataset(thm_data, OUTPUT_PATH)

if __name__ == "__main__":
    main()