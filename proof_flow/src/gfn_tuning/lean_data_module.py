import json
import warnings
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from proof_flow.src.utils import (
    prepare_environment_for_lean_dojo,
    repo_root,
)


warnings.filterwarnings("ignore", ".*does not have many workers.*")
prepare_environment_for_lean_dojo()


from lean_dojo import LeanGitRepo, Pos, Theorem, is_available_in_cache # isort: skip

def custom_theorem_collate_fn(batch: list[Theorem]) -> list[Theorem]:
    return batch


class NTPDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: Optional[str] = None,
        train_size: float = 0.95,
        train_data_path: Optional[str] = None,
        val_data_path: Optional[str] = None,
        repeat_theorem_n_times: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_data = None
        self.val_data = None
        self.repeat_n_times = repeat_theorem_n_times

    def setup(self, stage: str):
        assert (
            self.hparams.data_path is not None
            or (
                self.hparams.train_data_path is not None
                and self.hparams.val_data_path is not None
            )
        ), "data_path OR (train_data_path AND val_data_path) required"
        if self.hparams.data_path is not None:
            # case 1: single data_path to be split into train and val
            theorems = self._get_theorems_from_file(self.hparams.data_path)
            # split theorems into train and val
            if isinstance(self.hparams.train_size, float):
                num_train = int(len(theorems) * self.hparams.train_size)
            else:
                num_train = self.hparams.train_size
            train_theorems = self._repeat_theorems(theorems[:num_train])
            self.train_data = TheoremDataset(train_theorems)
            self.val_data = TheoremDataset(theorems[num_train:])
        else:
            # case 2: train_data_path and val_data_path are provided
            train_theorems = self._repeat_theorems(
                self._get_theorems_from_file(self.hparams.train_data_path)
            )
            self.train_data = TheoremDataset(train_theorems)
            self.val_data = TheoremDataset(
                self._get_theorems_from_file(self.hparams.val_data_path)
            )


    def train_dataloader(self):
        # data loader init args are copied from original code base
        # - batch_size=None (no batching), num_workers=0 (no new threads)
        return DataLoader(
            self.train_data, 
            shuffle=False, 
            batch_size=1, 
            num_workers=0,
            collate_fn=custom_theorem_collate_fn,
        )


    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=1, 
            num_workers=0,
            collate_fn=custom_theorem_collate_fn,
        )

    
    def _get_theorems_from_file(self, path: str) -> list[Theorem]:
        with open(repo_root() / path) as f:
            thm_dicts = json.load(f)
        thm0 = next(iter(thm_dicts.values()))
        repo = LeanGitRepo(thm0["url"], thm0["commit"])
        theorems = []
        for thm_dict in thm_dicts.values():
            thm = Theorem(repo, thm_dict["file_path"], thm_dict["full_name"])
            theorems.append(thm)
        return theorems
    

    def _repeat_theorems(self, theorems: list[Theorem]) -> list[Theorem]:
        if self.repeat_n_times == 1:
            return theorems
        repeated = []
        for t in theorems:
            for _ in range(self.repeat_n_times):
                repeated.append(t)
        return repeated


class TheoremDataset(Dataset):
    def __init__(self, theorems) -> None:
        super().__init__()
        self.theorems = theorems 

    def __len__(self):
        return len(self.theorems)

    def __getitem__(self, index):
        return self.theorems[index]
