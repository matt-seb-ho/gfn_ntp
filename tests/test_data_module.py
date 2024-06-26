import pytest

from src.constants import LEAN_DOJO_RANDOM_DATA_PATH
from src.gfn_tuning.lean_data_module import NTPDataModule, TheoremDataPipe
from src.utils import make_path_relative_to_repo, prepare_environment_for_lean_dojo

prepare_environment_for_lean_dojo()
from lean_dojo import LeanGitRepo, Pos, Theorem, is_available_in_cache # isort: skip

LIMIT_THEOREMS = 100
DATA_PATH = make_path_relative_to_repo(LEAN_DOJO_RANDOM_DATA_PATH)

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
        
    