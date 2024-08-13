import pytest

from proof_flow.src.constants import LEAN_DOJO_RANDOM_DATA_PATH
from proof_flow.src.gfn_tuning.lean_data_module import NTPDataModule, TheoremDataPipe
from proof_flow.src.utils import repo_root, prepare_environment_for_lean_dojo

prepare_environment_for_lean_dojo()
from lean_dojo import LeanGitRepo, Pos, Theorem, is_available_in_cache # isort: skip

LIMIT_THEOREMS = 100
DATA_PATH = repo_root() / LEAN_DOJO_RANDOM_DATA_PATH

def test_data_module():
    data = NTPDataModule(
        data_path=DATA_PATH,
        limit_theorems=LIMIT_THEOREMS,
    )
    data.setup("fit")
    train_data = data.train_dataloader()
    thm0 = next(iter(train_data))
    assert isinstance(thm0, Theorem)

def test_lean_dojo():
    data = NTPDataModule(data_path="single_thm.json",)
    train_data = data.train_dataloader()
    for thm in train_data:
        continue
    print(thm)
        
    