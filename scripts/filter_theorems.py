import argparse
import json
import os
import time
from tqdm import tqdm
from typing import Optional
import math
import numpy as np
from transformers import AutoTokenizer
# import matplotlib.pyplot as plt
# plt.style.use("rose-pine-dawn")

def filter_dataset(
    data_dir: str = "../data/",
    lean_dojo_split: str = "random",
    split: str = "train",
    tokenizer_id: str = "EleutherAI/llemma_7b",
    min_depth: int = 1,
    max_depth: int = 3,
    token_limit: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> tuple[list[int], list[int]]:
    """
    - creates a data subset with proofs of a maximum depth of `max_depth`.
    - saves the dataset to the provided path
    - gathers statistics about tactic token lengths
    """
    train_data_path = os.path.join(
        data_dir,
        "leandojo_benchmark_4",
        lean_dojo_split,
        f"{split}.json",
    )
    start = time.perf_counter()
    with open(train_data_path, "r") as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} proofs in {time.perf_counter() - start:.2f} seconds")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    filtered = []
    proof_lengths = []
    tactic_lengths = []
    token_limit = token_limit or math.inf
    
    print("Filtering proofs...")
    for proof in tqdm(train_data, total=len(train_data)):
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
    filename = f"{lean_dojo_split}_{split}_md{max_depth}_tl{token_limit}.json"
    output_path = os.path.join(output_dir or data_dir, filename)
    start = time.perf_counter()
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)
        print(f"Saved to: {output_path}")
        print(f"- {len(filtered)} proofs in {time.perf_counter() - start:.2f} seconds")
    
    return proof_lengths, tactic_lengths

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

def relative_to_repo(path):
    scripts_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(scripts_dir, "..", path))

if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument("--data_dir", "-dd", type=str, default="data/")
    psr.add_argument("--lean_dojo_split", "-ldsp", type=str, default="novel_premises")
    psr.add_argument("--split", "-sp", type=str, default="train")
    psr.add_argument("--tokenizer_id", "-tok", type=str, default="EleutherAI/llemma_7b")
    psr.add_argument("--output_path", "-out", type=str, default="data/")
    psr.add_argument("--min_depth", "-mind", type=int, default=1)
    psr.add_argument("--max_depth", "-maxd", type=int, default=3)
    psr.add_argument("--token_limit", "-maxt", type=int, default=None)
    args = psr.parse_args()
    
    # make data_dir path relative to the repo root
    
    proof_lengths, tactic_lengths = filter_dataset(
        data_dir=relative_to_repo(args.data_dir),
        lean_dojo_split=args.lean_dojo_split,
        split=args.split,
        tokenizer_id=args.tokenizer_id,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        token_limit=args.token_limit,
        output_dir=relative_to_repo(args.output_path),
    )

    print(
        "\n".join([
            "# Proof Lengths Statistics",
            summary_statistics(proof_lengths),
            deciles(proof_lengths),
            "------------------------------\n"
            "# Tactic Lengths Statistics",
            summary_statistics(tactic_lengths),
            deciles(tactic_lengths),
        ])
    )
