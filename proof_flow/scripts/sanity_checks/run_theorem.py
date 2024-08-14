import argparse
import json
import os
from time import perf_counter

from proof_flow.scripts.data_preprocessing.time_entry import (
    time_entry_with_timeout
)
from proof_flow.src.utils import repo_root

# comment this line to disable debug logs -> stderr
os.environ["DEBUG"] = "true"

# environment setup handled by time_entry
from lean_dojo import LeanGitRepo


def main():
    default_theorem_file = repo_root() / "data/random_train_pl1_3_tl30.json"
    psr = argparse.ArgumentParser()
    psr.add_argument("--input", type=str, default=str(default_theorem_file))
    psr.add_argument("--idx", nargs="+", default=["0"])
    psr.add_argument("--dojo_timeout", type=int, default=15, help="lean timeout (post-init)")
    psr.add_argument("--entry_timeout", type=int, default=5, help="timeout for entering theorem")
    args = psr.parse_args()
    
    repo = None
    with open(args.input, "r") as f:
        data = json.load(f)
    
    for string_idx in args.idx:
        idx = int(string_idx)
        thm_info = data[idx]
        if repo is None:
            repo = LeanGitRepo(thm_info["url"], thm_info["commit"])
        print(f"Running theorem: {thm_info['full_name']} (idx {idx}, file: {thm_info['file_path']})")
        entry_time, error, msg = time_entry_with_timeout(
            thm_info, 
            repo,
            args.dojo_timeout,
            args.entry_timeout, 
        )
        if error is None:
            print(f"-> entry time: {entry_time}")
        else:
            entry_time = args.entry_timeout if entry_time is None else entry_time
            print(f"-> t={entry_time}s, error:\n-> {msg}")


if __name__ == "__main__":
    main()
