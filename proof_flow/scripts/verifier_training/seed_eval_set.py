import argparse
from collections import defaultdict
import json
from typing import Optional
from datasets import Dataset, DatasetDict

from proof_flow.src.utils import get_config, repo_root
from proof_flow.scripts.verifier_training.format_sft_data import (
    preprocess_data,
    format_sft_dataset,
)


# step 1: determine what data to start with
# - really, we want the test set from SFT
# - unfortunately, SFT dataset does not include theorem idx/name/file
# - fortunately, I used a fixed seed for the train-test split
# - so we can reconstruct the test set from the train set (this time with idx/name/file)

def get_split_idx_assignment(
    config_name: str = "verifier_training",
    output_file: Optional[str] = "data/sft_split_idxs.json",
) -> dict:
    # load config, we just need raw data path
    config = get_config(config_name=config_name)
    
    # reconstruct the splits, but with additional information
    records = preprocess_data(repo_root() / config.sft.data.raw_data)
    dataset = format_sft_dataset(
        records, 
        repo_root() / config.sft.data.formatted_dataset_dir, 
        train_size=config.sft.data.train_size, 
        include_next_state=config.sft.data.include_next_state,
        include_theorem_info=True,
    )

    # gather idxs (theorem idx, tactic idx)
    split_idxs = defaultdict(list)
    for split, data in dataset.items():
        for row in data:
            split_idxs[split].append((row["idx"], row["tac_idx"]))
    
    # save to disk and return
    if output_file is not None:
        with open(repo_root() / output_file, "w") as f:
            json.dump(split_idxs, f, indent=2)
    return split_idxs
    

def get_seed_theorems_for_eval(
    split_idxs: Optional[dict] = None,
    split_idxs_file: Optional[str] = "data/sft_split_idxs.json",
    split: str = "test",
    output_file: Optional[str] = "data/eval_seed_thms.json",
    config_name: str = "verifier_training",
) -> dict[str, dict]:
    # load split idxs
    assert split_idxs is not None or split_idxs_file is not None, "Provide split idxs"
    if split_idxs is None:
        with open(repo_root() / split_idxs_file, "r") as f:
            split_idxs = json.load(f)
    
    # load raw theorem file
    config = get_config(config_name=config_name)
    raw_data_path = repo_root() / config.sft.data.raw_data
    with open(raw_data_path, "r") as f:
        raw_data = json.load(f)

    # get the seed theorem idxs
    # - we get the idxs we want from split_idxs[split]
    # - ideally we should care that the idxs we grab from there are NOT in train split
    # - but this probably results in a smaller set
    other_idxs = set()
    for current_split, idxs in split_idxs.items():
        if current_split != split:
            other_idxs.update([idx for idx, _ in idxs])
    
    loose_idxs = set([idx for idx, _ in split_idxs[split]])
    strict_idxs = loose_idxs - other_idxs
    print(f"loose idxs: {len(loose_idxs)}, strict idxs: {len(strict_idxs)}")

    # get, save, and return the seed theorems
    seed_theorems = {}
    for idx in loose_idxs:
        thm_info = raw_data[idx]
        thm_info["eval_set_eligible"] = "strict" if idx in strict_idxs else "loose"
        seed_theorems[idx] = thm_info
    with open(repo_root() / output_file, "w") as f:
        json.dump(seed_theorems, f, indent=2)   
    return seed_theorems


if __name__ == "__main__":
    # get split idxs
    split_idxs = get_split_idx_assignment()
    
    # get seed theorems for eval
    get_seed_theorems_for_eval(split_idxs=split_idxs)
