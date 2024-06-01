import json
import os
from pathlib import Path
from time import perf_counter

from load_gh_token import load_github_access_token
load_github_access_token()

start = perf_counter()
from lean_dojo import (
    InitOptimizedDojo, 
    Theorem, 
    LeanGitRepo,
    ProofFinished,
    TacticState
)
print(f"imported from lean_dojo in {perf_counter() - start}s")

# assuming __file__ is in gfn_ntp/scripts/
project_root = Path(__file__).parents[1]
benchmark_splits = "random" # or "novel_premises"
split = "val" # or "train", "test"
theorem_file_path = (
    project_root
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
    start = perf_counter()
    mathlib_repo = LeanGitRepo(ttd["url"], ttd["commit"])
    print(f"constructed LeanGitRepo in {perf_counter() - start}s")
    test_theorem = Theorem(mathlib_repo, ttd["file_path"], ttd["full_name"])
    
    # initialize repo for dojo
    # 1. downloads repo copy into tmp
    # 2. downloads traced repo into cache
    start = perf_counter()
    tmp_dir = project_root / "tmp"
    InitOptimizedDojo.init_repo(mathlib_repo, tmp_dir)
    print(f"downloaded repo and traced version in {perf_counter() - start}s")
    # - downloading both initially takes 170s
    # - subsequent runs only takes .0001s to check that it's there

    start = perf_counter()
    InitOptimizedDojo.default_tmp_dir = tmp_dir
    with InitOptimizedDojo(test_theorem, hard_timeout=30) as (dojo, initial_state):
        print(f"entered dojo in {perf_counter() - start}s")
        time_tactics(dojo, initial_state, ttd["traced_tactics"])
        

if __name__ == "__main__":
    main()
