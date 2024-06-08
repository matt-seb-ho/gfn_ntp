import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from enum import Enum
from typing import Optional

from tqdm import tqdm

from load_gh_token import load_github_access_token
load_github_access_token()

start = perf_counter()
from lean_dojo import (
    InitOptimizedDojo, 
    Theorem, 
    LeanGitRepo,
    ProofFinished,
    TacticState,
    DojoHardTimeoutError,
)
print(f"imported from lean_dojo in {perf_counter() - start}s")

# assuming __file__ is in gfn_ntp/scripts/
project_root = Path(__file__).parents[1]


class TimeDojoError(Enum):
    MISMATCH_BEFORE = 1
    MISMATCH_AFTER = 2
    SIMULATED_PROOF_UNFINISHED = 3
    TRACED_PROOF_UNFINISHED = 4
    BOTH_PROOFS_UNFINISHED = 5
    OTHER_TACTICS_ERROR = 6
    OTHER_ENTRY_ERROR = 7
    HARD_TIMEOUT = 8


def time_tactics(dojo, initial_state, tacs) -> tuple[list[int], Optional[TimeDojoError]]:
    # returns
    tactic_times = []
    state = initial_state
    for tt_ in tacs:
        if state.pp != tt_["state_before"]:
            return tactic_times, TimeDojoError.MISMATCH_BEFORE
        start = perf_counter()
        res = dojo.run_tac(state, tt_["tactic"])
        tactic_times.append(perf_counter() - start)
        if isinstance(res, TacticState):
            if res.pp != tt_["state_after"]:
                return tactic_times, TimeDojoError.MISMATCH_AFTER
        else:
            if not isinstance(res, ProofFinished):
                if tt_["state_after"] != "no goals":
                    return tactic_times, TimeDojoError.BOTH_PROOFS_UNFINISHED
                return tactic_times, TimeDojoError.SIMULATED_PROOF_UNFINISHED
            if tt_["state_after"] != "no goals":
                return tactic_times, TimeDojoError.TRACED_PROOF_UNFINISHED
        state = res
    return tactic_times, None

    
def time_theorems(thm_dicts, repo, output_dir, timeout, save_every=500, suffix=""):
    errors = defaultdict(list)
    tactics_times = []
    proof_times = []
    output_dir = Path(output_dir)

    suf = f"_{suffix}" if suffix else ""
    # helper routine to write progress to file
    def save_output(output_file, thm_dicts, errors, tactics_times, proof_times):
        # skip double counting OTHER_TACTICS_ERROR and OTHER_ENTRY_ERROR
        total_error_count = sum(
            len(v) for k, v in errors.items() 
            if k not in {TimeDojoError.OTHER_TACTICS_ERROR, TimeDojoError.OTHER_ENTRY_ERROR}
        )
        key_to_str = lambda x: x.name if isinstance(x, Enum) else str(x)
        errors_with_name_keys = {key_to_str(k): v for k, v in errors.items()}
        res = {
            "theorems": thm_dicts,
            "errors": errors_with_name_keys,
            "tactics_times": tactics_times, # all tactic times
            "proof_times": proof_times,     # sum of tactic times for each theorem
            "total_error_count": total_error_count,
        }
        with open(output_file, "w") as f:
            json.dump(res, f, indent=2)
        return res

    thms_processed = 0
    for idx, thm_info in tqdm(thm_dicts.items(), total=len(thm_dicts)):
        try:
            thm = Theorem(repo, thm_info["file_path"], thm_info["full_name"])
            start = perf_counter()
            with InitOptimizedDojo(thm, hard_timeout=timeout) as (dojo, initial_state):
                thm_info["entry_time"] = perf_counter() - start
                try:
                    tac_times, error = time_tactics(dojo, initial_state, thm_info["traced_tactics"])
                    tactics_times.extend(tac_times)
                    proof_times.append(sum(tac_times))
                    if error is not None:
                        errors[error].append(idx)
                except Exception as e:
                    errors[TimeDojoError.OTHER_TACTICS_ERROR].append(idx)
                    errors[str(e)].append(idx)
        except DojoHardTimeoutError:
            thm_info["entry_time"] = timeout
            errors[TimeDojoError.HARD_TIMEOUT].append(idx)
        except Exception as e:
            errors[TimeDojoError.OTHER_ENTRY_ERROR].append(idx)
            errors[str(e)].append(idx)
        # routinely save results in case of crash
        thms_processed += 1
        if thms_processed % save_every == 0:
            save_output(output_dir / f"intermediate{thms_processed}{suf}.json", thm_dicts, errors, tactics_times, proof_times)
    
    return save_output(output_dir / f"final{suf}.json", thm_dicts, errors, tactics_times, proof_times)


def main():
    psr = argparse.ArgumentParser()
    psr.add_argument("-n", type=int, default=1000, help="how many theorems to time entry")
    psr.add_argument("--input", type=str, default="/mnt/hdd/msho/gfn_ntp/data/random_train_pl1_3_tl30.json", help="theorems to time")
    psr.add_argument("--timeout", type=int, default=60, help="timeout for each theorem")
    psr.add_argument("--suffix", help="suffix to add to output filename")
    args = psr.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    
    # randomly select args.n theorems to time
    random.seed(42)
    idxs_to_test = random.sample(range(len(data)), args.n)
    theorems = {i: data[i] for i in idxs_to_test}
    
    # setup
    # - construct LeanGitRepo
    start = perf_counter()
    example_theorem = data[0]
    mathlib_repo = LeanGitRepo(example_theorem["url"], example_theorem["commit"])
    print(f"constructed LeanGitRepo in {perf_counter() - start}s")
    # - set tmp_dir
    InitOptimizedDojo.default_tmp_dir = project_root / "tmp"

    # timing code
    time_theorems(
        theorems, 
        mathlib_repo, 
        # project_root / f"data/dojo_times_n{args.n}_{args.suffix}.json", 
        project_root / "data/timing",
        args.timeout,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
