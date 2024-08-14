import json
import os
from pathlib import Path
from time import perf_counter

from proof_flow.src.utils import prepare_environment_for_lean_dojo, repo_root

prepare_environment_for_lean_dojo()

start = perf_counter()
from lean_dojo import ( # isort: skip
    Dojo, 
    Theorem, 
    LeanGitRepo,
    ProofFinished,
    TacticState
)
print(f"imported from lean_dojo in {perf_counter() - start}s")

benchmark_splits = "random" # or "novel_premises"
split = "val" # or "train", "test"
theorem_file_path = (
    repo_root()
    / "data/leandojo_benchmark_4"
    / benchmark_splits
    / f"{split}.json"
)


def time_tactics(dojo, initial_state, tacs):
    state = initial_state
    print("timing tactic run times")
    for i, tt_ in enumerate(tacs):
        assert state.pp == tt_["state_before"]
        start = perf_counter()
        res = dojo.run_tac(state, tt_["tactic"])
        print(f"tactic #{i+1} latency: {perf_counter() - start}s")
        if isinstance(res, TacticState):
            assert res.pp == tt_["state_after"]
        else:
            assert isinstance(res, ProofFinished) and tt_["state_after"] == "no goals"
        state = res


def main():
    with open(theorem_file_path) as f:
        data = json.load(f)
    ttd = data[0] # test theorem dict

    # initialize repo for dojo
    # 1. LeanGitRepo.__post_init__ downloads repo copy into tmp
    start = perf_counter()
    mathlib_repo = LeanGitRepo(ttd["url"], ttd["commit"])
    print(f"constructed LeanGitRepo in {perf_counter() - start}s")
    test_theorem = Theorem(mathlib_repo, ttd["file_path"], ttd["full_name"])

    # 2. Dojo.__enter__ downloads traced repo from remote cache into local cache
    # - downloading both initially takes ~170s
    # - subsequent runs only take .0001s to read from cache
    start = perf_counter()
    with Dojo(test_theorem, timeout=30) as (dojo, initial_state):
        print(f"entered dojo in {perf_counter() - start}s")
        time_tactics(dojo, initial_state, ttd["traced_tactics"])

        

if __name__ == "__main__":
    main()