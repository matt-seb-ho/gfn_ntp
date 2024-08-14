import argparse
import json
from time import perf_counter

import numpy as np
from tqdm import tqdm

from proof_flow.src.utils import prepare_environment_for_lean_dojo

prepare_environment_for_lean_dojo()
from lean_dojo import Dojo, Theorem, LeanGitRepo # isort: skip


if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument("--seed", type=int, default=42)
    psr.add_argument("--samples", type=int, default=30)
    psr.add_argument("--output", type=int, default=30)
    args = psr.parse_args()

    with open("data/leandojo_benchmark_4/novel_premises/train.json") as f:
        thm_dicts = json.load(f)

    np.random.seed(args.seed)
    idxs = np.random.choice(range(len(thm_dicts)), args.samples, replace=False)
    entry_times = []
    exit_times = []

    for idx in tqdm(idxs, total=args.samples):
        thm_dict = thm_dicts[idx] 
        thm = Theorem(
            repo=LeanGitRepo(url=thm_dict["url"], commit=thm_dict["commit"]),
            file_path=thm_dict["file_path"],
            full_name=thm_dict["full_name"]
        )
        start = perf_counter()
        with Dojo(thm) as (dojo, initial_state):
            entry_times.append(perf_counter() - start)
            start = perf_counter()
        exit_times.append(perf_counter() - start)

    with open(args.output_path, 'w') as f:
        json.dump({"idxs": list(idxs), "entry": entry_times, "exit": exit_times}, f, indent=2)
        print("wrote to outputs/entry_exit_times.json")
