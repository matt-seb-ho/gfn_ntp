import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
import shutil
from time import perf_counter
from enum import Enum
from typing import Optional
from multiprocessing import Process, Queue
import time

from tqdm import tqdm

from proof_flow.src.utils import prepare_environment_for_lean_dojo, repo_root
prepare_environment_for_lean_dojo()

start = perf_counter()
from lean_dojo import ( # isort: skip
    Dojo,
    Theorem, 
    LeanGitRepo,
    ProofFinished,
    TacticState,
    DojoHardTimeoutError,
    LeanError,
)
print(f"imported from lean_dojo in {perf_counter() - start}s")


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


class TimeoutException(Exception):
    pass


def dojo_enter_wrapper(theorem, dojo_timeout, queue):
    start = perf_counter()
    stop = None
    try:
        with Dojo(theorem, timeout=dojo_timeout) as (dojo, initial_state):
            stop = perf_counter()
        queue.put((stop - start, None))
    except Exception as e:
        stop = perf_counter()
        queue.put((stop - start, e))


def time_entry_with_timeout(
    thm_info: dict, 
    repo: LeanGitRepo, 
    dojo_timeout: int, 
    timing_timeout: int,
) -> tuple[Optional[float], Optional[TimeDojoError], str]:
    queue = Queue()
    thm = Theorem(repo, thm_info["file_path"], thm_info["full_name"])

    # ensure backup is created
    # thm_file_path = tmp_dir / repo.name / thm.file_path
    # manual_backup = thm_file_path.with_suffix(".backup")
    # shutil.copy(thm_file_path, manual_backup)
    
    proc = Process(target=dojo_enter_wrapper, args=(thm, dojo_timeout, queue))
    proc.start()
    proc.join(timing_timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        res = None, TimeDojoError.ENTRY_TIMEOUT, "entry timeout"
    else:
        entry_time, error = queue.get()
        if error is None:
            res = entry_time, None, ""
        else:
            res = entry_time, TimeDojoError.OTHER_ENTRY_ERROR, str(error)

    # ensure backup is restored
    # os.rename(manual_backup, thm_file_path)

    # return entry time, exception
    return res


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
    
"""
def time_theorem(
    thm: Theorem, 
    time_limit: HardTimeLimit,
    traced_tactics: list[dict],
    dojo_timeout: int = 60, 
) -> tuple[Optional[int], list[int], Optional[TimeDojoError], str]:
    # returns entry_time, tactic_times, error, error message
"""


def time_theorems(
    thm_dicts: dict[int, dict],
    repo: LeanGitRepo, 
    output_dir: str, 
    dojo_timeout: int = 10, 
    save_every: int = 500, 
    suffix: str = "",
    entry_timeout: int = 5,
) -> dict[str, dict]:
    errors = defaultdict(list)
    entry_times = {}
    output_dir = Path(output_dir)
    suffix = f"_{suffix}" if suffix else ""

    # helper routine to write progress to file
    def save_output(output_file):
        key_to_str = lambda x: x.name if isinstance(x, Enum) else str(x)
        json_friendly_errors = {key_to_str(k): v for k, v in errors.items()}
        res = {
            "theorems": thm_dicts,
            "entry_times": entry_times,
            "errors": json_friendly_errors,
        }
        with open(output_file, "w") as f:
            json.dump(res, f, indent=2)
        return res

    for i, (thm_idx, thm_info) in enumerate(tqdm(thm_dicts.items())):
        try:
            entry_time, error, error_msg = time_entry_with_timeout(
                thm_info, 
                repo, 
                dojo_timeout, 
                entry_timeout,
            )

            # record results
            entry_times[thm_idx] = entry_time
            thm_info["entry_time"] = entry_time
            thm_info["entry_failed"] = error is not None
            if error is not None:
                errors[error].append((thm_idx, error_msg))

            # routinely save results in case of crash
            if i != 0 and i % save_every == 0:
                save_output(output_dir / f"intermediate{i}{suffix}.json")

        except Exception as e:
            print(f"Error timing theorem {thm_idx} (iter {i}): {type(e)}: {e}")
            entry_times[thm_idx] = None
            thm_info["entry_time"] = None
            thm_info["entry_failed"] = True
            errors[TimeDojoError.OTHER_ENTRY_ERROR].append((thm_idx, str(e)))
    
    # print summary
    total_error_count = sum(len(v) for v in errors.values())
    print(f"total errors: {total_error_count}, total theorems: {len(thm_dicts)}")
    print("error types:")
    for k, v in errors.items():
        print(f"{k.name}: {len(v)}")

    # save final result
    return save_output(output_dir / f"final{suffix}.json")


def main():
    psr = argparse.ArgumentParser()
    psr.add_argument("--n", type=int, default=1000, help="how many theorems to time entry, 0 for all")
    psr.add_argument("--input", type=str, default="/mnt/hdd/msho/gfn_ntp/data/random_train_pl1_3_tl30.json", help="theorems to time")
    psr.add_argument("--dojo_timeout", type=int, default=60, help="timeout for each theorem")
    psr.add_argument("--suffix", help="suffix to add to output filename")
    psr.add_argument("--entry_timeout", type=int, default=5, help="timeout for entering theorem")
    psr.add_argument("--skip_first", type=int, help="continuing a failed run")
    psr.add_argument("--test_run", action="store_true")
    args = psr.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    
    # randomly select args.n theorems to time
    if args.test_run:
        # expecting 3965 to be ~1.7s, 8108 to be ~20s (timeout error)
        # idxs_to_test = [3965, 8108]
        # err2s from v2 (file not founds)
        idxs_to_test = [20989, 19216, 25936, 25418, 18805, 15458, 25980, 13436, 2782, 15271]
    elif args.n == 0:
        idxs_to_test = range(len(data))
    else:
        random.seed(42)
        idxs_to_test = random.sample(range(len(data)), args.n)
        if args.skip_first:
            idxs_to_test = idxs_to_test[args.skip_first:]
    theorems = {i: data[i] for i in idxs_to_test}
    
    # setup
    # - construct LeanGitRepo
    start = perf_counter()
    example_theorem = data[0]
    mathlib_repo = LeanGitRepo(example_theorem["url"], example_theorem["commit"])
    print(f"constructed LeanGitRepo in {perf_counter() - start}s")

    # timing code
    time_theorems(
        theorems, 
        mathlib_repo, 
        repo_root() / "data/timing",
        dojo_timeout=args.dojo_timeout,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
