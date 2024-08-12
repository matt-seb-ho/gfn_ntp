import os
os.environ["GITHUB_ACCESS_TOKEN"] = "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"
ghat = os.getenv("GITHUB_ACCESS_TOKEN", None)
print(f"ghat: {ghat}")

from lean_dojo import LeanGitRepo, Theorem, Pos, is_available_in_cache
from src.gfn_tuning.lean_data_module import NTPDataModule, TheoremDataPipe
from src.gfn_tuning.constants import LEAN_DOJO_RANDOM_DATA_PATH 


def test_data_module():
    print(f"data_path: {LEAN_DOJO_RANDOM_DATA_PATH}")
    data = NTPDataModule(
        data_path=LEAN_DOJO_RANDOM_DATA_PATH,
        limit_theorems=100,
    )
    data.setup("fit")
    train_data = data.train_dataloader()
    thm0 = next(iter(train_data))
    print(thm0)
    assert isinstance(thm0, Theorem)
    
def main():
    test_data_module()

if __name__ == '__main__':
    main()
