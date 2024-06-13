import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from enum import Enum
from typing import Optional
import signal

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
    LeanError,
)
print(f"imported from lean_dojo in {perf_counter() - start}s")

# assuming __file__ is in gfn_ntp/scripts/
project_root = Path(__file__).parents[1]


class HardTimeLimit:
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.timed_out = False
    
    def start(self):
        signal.signal(signal.SIGALRM, self.alarm_handler)
        signal.alarm(self.time_limit)
    
    def cancel(self):
        signal.alarm(0) # disable alarm
        signal.signal(signal.SIGALRM, signal.SIG_DFL) # reset signal handler
    
    def alarm_handler(self, signum, frame):
        raise HitEntryTimeLimit()


class HitEntryTimeLimit(Exception):
    pass


class TimeDojoError(Enum):
    INCOMPLETE_PROOF = 1
    LEAN_ERROR = 2
    EARLY_FINISH = 3
    ENTRY_TIMEOUT = 4
    DOJO_TIMEOUT = 5
    OTHER_ENTRY_ERROR = 6
    OTHER_TACTIC_ERROR = 7


def time_tactics(dojo, initial_state, tacs) -> tuple[list[int], Optional[TimeDojoError]]:
    tactic_times = []
    state = initial_state
    for i, tt_ in enumerate(tacs):
        start = perf_counter()
        tactic_result = dojo.run_tac(state, tt_["tactic"])
        tactic_times.append(perf_counter() - start)

        # v2 update: only interested if (1) tactic failed (2) proof finished early (3) proof never finished
        if isinstance(tactic_result, LeanError):
            return tactic_times, TimeDojoError.LEAN_ERROR
        elif isinstance(tactic_result, ProofFinished) and i < len(tacs) - 1:
            return tactic_times, TimeDojoError.EARLY_FINISH
        state = tactic_result

        ### v1 code (used for initial n=500, 4096)
        #
        # if state.pp != tt_["state_before"]:
        #     return tactic_times, TimeDojoError.MISMATCH_BEFORE
        #
        # start = perf_counter()
        # res = dojo.run_tac(state, tt_["tactic"])
        # tactic_times.append(perf_counter() - start)
        #
        # if isinstance(res, TacticState):
        #     if res.pp != tt_["state_after"]:
        #         return tactic_times, TimeDojoError.MISMATCH_AFTER
        # else:
        #     if not isinstance(res, ProofFinished):
        #         # if tt_["state_after"] != "no goals":
        #         #     return tactic_times, TimeDojoError.BOTH_PROOFS_UNFINISHED
        #         return tactic_times, TimeDojoError.SIMULATED_PROOF_UNFINISHED
        #     if tt_["state_after"] != "no goals":
        #         return tactic_times, TimeDojoError.TRACED_PROOF_UNFINISHED
        #
        # state = res
    
    if not isinstance(state, ProofFinished):
        return tactic_times, TimeDojoError.INCOMPLETE_PROOF
    return tactic_times, None

    
def entry_timeout_handler(signum, frame):
    raise HitEntryTimeLimit
    
def time_theorem(
    thm: Theorem, 
    time_limit: HardTimeLimit,
    traced_tactics: list[dict],
    dojo_timeout: int = 60, 
) -> tuple[Optional[int], list[int], Optional[TimeDojoError], str]:
    # returns entry_time, tactic_times, error, error message
    try:
        time_limit.start()
        start = perf_counter()
        with InitOptimizedDojo(thm, hard_timeout=dojo_timeout) as (dojo, initial_state):
            entry_time = perf_counter() - start
            time_limit.cancel()
            try:
                tac_times, error = time_tactics(dojo, initial_state, traced_tactics)
                return entry_time, tac_times, error, ""
            except Exception as e:
                return entry_time, [], TimeDojoError.OTHER_TACTIC_ERROR, str(e)
    except HitEntryTimeLimit:
        return None, [], TimeDojoError.ENTRY_TIMEOUT, ""
    except DojoHardTimeoutError:
        return None, [], TimeDojoError.DOJO_TIMEOUT, ""
    except Exception as e:
        return None, [], TimeDojoError.OTHER_ENTRY_ERROR, str(e)


def time_theorems(
    thm_dicts, 
    repo, 
    output_dir, 
    dojo_timeout, 
    save_every=500, 
    suffix="",
    entry_timeout=5,
):
    errors = defaultdict(list)
    tactics_times = []
    proof_times = []
    entry_times = {}
    output_dir = Path(output_dir)
    suffix = f"_{suffix}" if suffix else ""

    # helper routine to write progress to file
    def save_output(output_file):
        key_to_str = lambda x: x.name if isinstance(x, Enum) else str(x)
        json_friendly_errors = {key_to_str(k): v for k, v in errors.items()}
        res = {
            "theorems": thm_dicts,
            "errors": json_friendly_errors,
            "tactics_times": tactics_times, # all tactic times
            "proof_times": proof_times,     # sum of tactic times for each theorem
            "entry_times": entry_times,     # time to enter each theorem
        }
        with open(output_file, "w") as f:
            json.dump(res, f, indent=2)
        return res

    time_limit = HardTimeLimit(entry_timeout)
    for i, (thm_idx, thm_info) in enumerate(tqdm(thm_dicts.items())):
        # construct theorem object
        thm = Theorem(repo, thm_info["file_path"], thm_info["full_name"])
        # time theorem entry and try running tactics
        entry_time, tac_times, error, error_msg = time_theorem(
            thm,
            time_limit,
            thm_info["traced_tactics"],
            dojo_timeout=dojo_timeout,
        )
        # record results
        entry_times[thm_idx] = entry_time
        thm_info["entry_time"] = entry_time
        thm_info["entry_failed"] = (
            error == TimeDojoError.ENTRY_TIMEOUT
            or error == TimeDojoError.OTHER_ENTRY_ERROR
        )
        tactics_times.extend(tac_times)
        if error is None or error == TimeDojoError.EARLY_FINISH:
            proof_times.append(sum(tac_times))
        elif error is not None:
            errors[error].append((thm_idx, error_msg))

        # routinely save results in case of crash
        if i != 0 and i % save_every == 0:
            save_output(output_dir / f"intermediate{i}{suffix}.json")
    
    return save_output(output_dir / f"final{suffix}.json")


def main():
    psr = argparse.ArgumentParser()
    psr.add_argument("-n", type=int, default=1000, help="how many theorems to time entry")
    psr.add_argument("--input", type=str, default="/mnt/hdd/msho/gfn_ntp/data/random_train_pl1_3_tl30.json", help="theorems to time")
    psr.add_argument("--timeout", type=int, default=60, help="timeout for each theorem")
    psr.add_argument("--suffix", help="suffix to add to output filename")
    psr.add_argument("--entry_timeout", type=int, default=5, help="timeout for entering theorem")
    psr.add_argument("--test_run", action="store_true")
    args = psr.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    
    # randomly select args.n theorems to time
    if args.test_run:
        # expecting 3965 to be ~1.7s, 8108 to be ~20s (timeout error)
        idxs_to_test = [3965, 8108]
    else:
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
