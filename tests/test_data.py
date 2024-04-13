import pytest
from lean_dojo import LeanGitRepo, Pos, Theorem, is_available_in_cache

from src.constants import LEAN_DOJO_RANDOM_DATA_PATH
from src.gfn_tuning.lean_data_module import NTPDataModule, TheoremDataPipe

LIMIT_THEOREMS = 100
DATA_PATH = LEAN_DOJO_RANDOM_DATA_PATH

def test_data_module():
    data = NTPDataModule(
        data_path=DATA_PATH,
        limit_theorems=LIMIT_THEOREMS,
    )
    data.setup("fit")
    train_data = data.train_dataloader()
    thm0 = next(iter(train_data))
    assert isinstance(thm0, Theorem)
