import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from lean_dojo import LeanGitRepo
# time_entry handles GH auth token
from proof_flow.scripts.data_preprocessing.formal_statements import (
    add_formal_statements,
)
from proof_flow.scripts.data_preprocessing.time_entry import time_theorems
from tqdm import tqdm
from transformers import AutoTokenizer

from proof_flow.src.utils import repo_root

data_dir = repo_root() / "data"


def filter_dataset_for_init_time(
    threshold: int,
    timing_files: list[str],
    output_file: str,
    thm_dicts: Optional[dict[int, dict]] = None,
    **timing_kwargs
) -> dict[int, dict]:
    """
    Filter theorems based on initialization time.

    input:
    - either provide timing files from `time_entry.py:time_theorems()`
    - or {idx: thm_dict} to time theorems from scratch (provide time_theorems() kwargs)
    output:
    - filtered {idx: thm_dict}

    """
    if len(timing_files):
        # gather timing data
        timed_theorems = {}
        for timing_file in timing_files:
            with open(timing_file, "r") as f:
                timing_data = json.load(f)
                timed_theorems.update(timing_data["theorems"])
        print(f"timed theorems count: {len(timed_theorems)}")
    else:
        # get new timing data
        assert thm_dicts is not None, "Please provide theorems to time"
        thm0 = next(iter(thm_dicts.values()))
        repo = LeanGitRepo(thm0["url"], thm0["commit"])
        timing_data = time_theorems(
            thm_dicts,
            repo,
            timing_kwargs.get("output_dir", data_dir / "timing"),
            **timing_kwargs,
        )
        timed_theorems = timing_data["theorems"]
    
    # gather theorems that initialized successfully and within the time threshold
    filtered_theorems = {}
    exclusion_stats = {
        "entry_failure": 0,
        "threshold": 0,
        "no_timing_info": 0,
    }
    
    for idx, thm_info in timed_theorems.items():
        if "entry_time" not in thm_info and "entry_failed" not in thm_info:
            # case 1: lacking timing info (skip)
            exclusion_stats["no_timing_info"] += 1
            continue
        elif (thm_info.get("entry_time", None) or -1) > threshold:
            # case 2: entry time exceeds threshold (skip)
            exclusion_stats["threshold"] += 1
            continue
        elif thm_info.get("entry_failed", False):
            # case 3: entry otherwise failed (skip)
            exclusion_stats["entry_failure"] += 1
            continue
        # case 4: passes time criteria (include in filtered results)
        filtered_theorems[idx] = thm_info

    print(f"filtered theorems count: {len(filtered_theorems)}")
    print("excluded theorem stats:", json.dumps(exclusion_stats, indent=2))
    
    # save the filtered data
    with open(output_file, "w") as f:
        json.dump(filtered_theorems, f, indent=2)
        print(f"finished dojo entry time filter; saved {len(filtered_theorems)} to: {output_file}")
    
    return filtered_theorems
        
        
def filter_out_top5_percent_tokens(iput_data_path: str, output_data_path: str, tokenizer_id: str="EleutherAI/llemma_7b"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    with open(iput_data_path) as f:
        data = json.load(f)

    data_k_v = data.items()
    num_entries_to_remove = int(len(data) * 0.05)

    def count_tokens(text):
        return len(tokenizer.tokenize(text))

    sorted_data = sorted(
        data_k_v, 
        key=lambda x: count_tokens(x[1]["traced_tactics"][0]["state_before"]), 
        reverse=True
    )

    keys_to_remove = [key for key, _ in sorted_data[:num_entries_to_remove]]
    for key in keys_to_remove:
        del data[key]
        
    with open(output_data_path, "w") as f:
        data = json.dump(data, f, indent=4)


def filter_dataset_for_length(
    data_file_path: Optional[str] = None,
    data_dir: str = data_dir,
    benchmark_splits: str = "random", # or "novel_premises"
    split: str = "train", # or "val", "test"
    tokenizer_id: str = "EleutherAI/llemma_7b",
    min_depth: int = 1,
    max_depth: int = 3,
    token_limit: Optional[int] = None,
    output_file: Optional[str] = None,
) -> tuple[list[int], list[int]]:
    """
    - creates a data subset with proofs of a maximum depth of `max_depth`.
    - saves the dataset to the provided path
    - gathers statistics about tactic token lengths
    """

    full_data_path = data_file_path or os.path.join(
        data_dir,
        "leandojo_benchmark_4",
        benchmark_splits,
        f"{split}.json",
    )
    start = time.perf_counter()
    with open(full_data_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} proofs in {time.perf_counter() - start:.2f} seconds")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    filtered = []
    proof_lengths = []
    tactic_lengths = []
    token_limit = token_limit or math.inf
    
    print("Filtering proofs...")
    for proof in tqdm(data, total=len(data)):
        proof_length = len(proof["traced_tactics"])
        proof_lengths.append(proof_length)
        if not (min_depth <= proof_length <= max_depth):
            continue

        token_filter_status = True
        for traced_tactic in proof["traced_tactics"]:
            tactic = traced_tactic["tactic"]
            token_length = len(tokenizer.encode(tactic))
            tactic_lengths.append(token_length)
            if token_length > token_limit:
                token_filter_status = False

        if token_filter_status:
            filtered.append(proof)
    
    # save the filtered data
    output_path = (
        output_file 
        or os.path.join(data_dir, f"{benchmark_splits}_{split}_pl{min_depth}_{max_depth}_tl{token_limit}.json")
    )
    with open(output_path, "w") as f:
        filtered_dict = {idx: thm for idx, thm in enumerate(filtered)}
        json.dump(filtered_dict, f, indent=2)
        print(f"finished initial length filter; saved {len(filtered)} entries to: {output_path}")
    
    return proof_lengths, tactic_lengths


def filter_dataset_for_tactics_style(
    thm_dicts_file: str,
    output_file: str,
    stats_file: Optional[str] = None,
):
    """
    Only keep theorems with proofs completely in tactic style.
    """
    with open(thm_dicts_file, "r") as f:
        thm_dicts = json.load(f)
    data, _ = add_formal_statements(thm_dicts, stats_file=stats_file)
    filtered_theorems = {}
    for idx, thm_info in data.items():
        if thm_info.get("is_tactic_proof", False):
            filtered_theorems[idx] = thm_info
    with open(output_file, "w") as f:
        json.dump(filtered_theorems, f, indent=2)
        print(f"finished proof style (tactics only) filter; saved {len(filtered_theorems)} entries to: {output_file}")
    

def summary_statistics(data):
    return (
        f"Mean: {np.mean(data):.2f}\n"
        f"Median: {np.median(data):.2f}\n"
        f"Standard Deviation: {np.std(data):.2f}\n"
        f"Variance: {np.var(data):.2f}\n"
        f"Min: {np.min(data):.2f}\n"
        f"Max: {np.max(data):.2f}\n"
        f"Range: {np.ptp(data):.2f}\n"
        f"25th percentile: {np.percentile(data, 25):.2f}\n"
        f"75th percentile: {np.percentile(data, 75):.2f}"
    )


def deciles(data):
    return "\n".join(
        f"{i}th percentile: {np.percentile(data, i):.2f}"
        for i in range(0, 100, 10)
    )


def main():
    psr = argparse.ArgumentParser()

    # filtering on length (proof steps, and tokens within each step)
    psr.add_argument("--filter_length", action="store_true", help="filter on proof step count and tokens per step")
    psr.add_argument("--output_file", type=str, default="filtered_theorems.json")

    # - specifying initial data
    psr.add_argument("--data_dir", default=str(data_dir), help="directory to read benchmark from and write filtered files to")
    psr.add_argument("--splits", choices=["random", "novel_premises"], default="random")
    psr.add_argument("--split", choices=["train", "val", "test"], default="train")
    psr.add_argument("--tokenizer", type=str, default="EleutherAI/llemma_7b")

    # - length filter parameters 
    psr.add_argument("--min_depth", type=int, default=1)
    psr.add_argument("--max_depth", type=int, default=3)
    psr.add_argument("--token_limit", type=int, default=None)

    # filtering on proof style
    psr.add_argument("--filter_tactic_proof", action="store_true", help="keep only proofs entirely in tactic style")
    psr.add_argument("--fs_stats_file", type=str, help="file to save formal statement extraction stats to")

    # filtering on entry time (TODO: set threshold default)
    psr.add_argument("--filter_time", action="store_true", help="filter on (real and estimated) entry time")
    psr.add_argument("--data_file", nargs="+", help="files containing timing data")
    psr.add_argument("--time_threshold", type=int, default=5, help="threshold for entry time (in seconds) to filter on")

    # misc options
    psr.add_argument(
        "--stats", 
        nargs='?', 
        const="data/filter_theorems_stats.json",
        default=None, 
        help="pass as flag to collect statistics and optionally specify stats file path",
    )
        
    args = psr.parse_args()
    fs_stats_file = repo_root() / args.fs_stats_file if args.fs_stats_file else None

    if args.filter_length:   
        proof_lengths, tactic_lengths = filter_dataset_for_length(
            data_dir=args.data_dir,
            benchmark_splits=args.splits,
            split=args.split,
            tokenizer_id=args.tokenizer,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            token_limit=args.token_limit,
            output_file=args.output_file,
        )
        # print length statistics
        print((
            "\n".join([
                "# Proof Lengths Statistics",
                summary_statistics(proof_lengths),
                deciles(proof_lengths),
                "------------------------------\n"
                "# Tactic Lengths Statistics",
                "  - note: only theorems with acceptable proof lengths are included",
                summary_statistics(tactic_lengths),
                deciles(tactic_lengths),
            ])
        ))
        # filter on proof style (tactics only)
        if args.filter_tactic_proof:
            filter_dataset_for_tactics_style(
                repo_root() / args.output_file,
                repo_root() / args.output_file,
                stats_file=fs_stats_file,
            )
    elif args.filter_tactic_proof:
        filter_dataset_for_tactics_style(
            repo_root() / args.data_file,
            repo_root() / args.output_file,
            stats_file=fs_stats_file,
        )
    elif args.filter_time:
        filter_dataset_for_init_time(
            args.time_threshold,
            [repo_root() / f for f in args.data_file],
            repo_root() / args.output_file
        )
        filter_out_top5_percent_tokens(
            repo_root() / args.output_file,
            repo_root() / args.output_file,
            tokenizer_id=args.tokenizer,
        )

if __name__ == "__main__":
    main()
