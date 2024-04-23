import json
from time import perf_counter
import numpy as np
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv("/home/matthewho/.env") # load GH token env var
from lean_dojo import Dojo, Theorem, LeanGitRepo

with open("data/leandojo_benchmark_4/novel_premises/train.json") as f:
    thm_dicts = json.load(f)

samples = 30
np.random.seed(42)
idxs = np.random.choice(range(len(thm_dicts)), samples, replace=False)
entry_times = []
exit_times = []

for idx in tqdm(idxs, total=samples):
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

with open("outputs/entry_exit_times.json", 'w') as f:
    json.dump({"idxs": list(idxs), "entry": entry_times, "exit": exit_times}, f, indent=2)
    print("wrote to outputs/entry_exit_times.json")
