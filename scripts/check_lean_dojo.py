import json
import os
from contextlib import contextmanager
from time import perf_counter

from lean_dojo_env_setup import prepare_environment_for_lean_dojo

prepare_environment_for_lean_dojo()
from lean_dojo import Dojo, LeanGitRepo, ProofFinished, TacticState, Theorem # isort: skip


def load_theorems_from_json(file, num_theorems=1, filter=None):
    print(f"loading data from path: {file}")
    with open(file) as f:
        data = json.load(f)
    theorems = []
    for theorem in data:
        if filter is None or filter(theorem):
            theorems.append(theorem)
        if num_theorems and (len(theorems) == num_theorems):
            return theorems
    return theorems

def load_theorems_from_benchmark(data_dir, ld_split="novel_premises", split="train", num_theorems=1):
    file_path = os.path.join(data_dir, "leandojo_benchmark_4", ld_split, split + ".json")
    load_theorems_from_json(
        file_path,
        num_theorems=num_theorems,
        filter=lambda e: len(e["traced_tactics"]) > 0
    )

def get_thm_from_file_entry(e):
    repo = LeanGitRepo(e["url"], e["commit"])
    thm = Theorem(repo=repo, file_path=e["file_path"], full_name=e["full_name"])
    return thm

@contextmanager
def timed_dojo(theorem):
    start = perf_counter()
    with Dojo(theorem) as (dojo, initial_state):
        entry_time = perf_counter() - start
        yield (dojo, initial_state)
        start = perf_counter()
    exit_time = perf_counter() - start
    print(f"dojo entry time: {entry_time:.2f}s, exit time: {exit_time:.2f}s")

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

if __name__ == "__main__":
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    wrt2repo = lambda p: os.path.realpath(os.path.join(scripts_dir, "..", p))
    # tts = load_theorems_from_benchmark(wrt2repo("data/"), num_theorems=2)
    tts = load_theorems_from_json(wrt2repo("data/novel_premises_train_md3_tlinf.json"), 2)
    tt = tts[0]
    theorem = get_thm_from_file_entry(tt)
    print("first theorem:", theorem)

    start = perf_counter()
    with timed_dojo(theorem) as (dojo, initial_state):
        time_tactics(dojo, initial_state, tt["traced_tactics"])

    with timed_dojo(get_thm_from_file_entry(tts[1])) as (dojo, initial_state):
        print("^ second theorem")

    with timed_dojo(get_thm_from_file_entry(tts[1])) as (dojo, initial_state):
        print("^ first theorem")
