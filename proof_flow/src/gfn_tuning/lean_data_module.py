import hashlib
import json
import os
import warnings
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe

from proof_flow.src.utils import (
    prepare_environment_for_lean_dojo,
    repo_root,
)


warnings.filterwarnings("ignore", ".*does not have many workers.*")
prepare_environment_for_lean_dojo()


from lean_dojo import LeanGitRepo, Pos, Theorem, is_available_in_cache # isort: skip


class NTPDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        train_size=0.95,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_data = None
        self.val_data = None

    def setup(self):
        # read theorem dicts from json file
        with open(repo_root() / self.hparams.data_path) as f:
            thm_dicts = json.load(f)
        # create a LeanGitRepo object and a list of Theorem objects
        thm0 = next(iter(thm_dicts))
        repo = LeanGitRepo(thm0["url"], thm0["commit"])
        theorems: list[Theorem] = []
        for thm_dict in thm_dicts:
            thm = Theorem(repo, thm_dict["file_path"], thm_dict["full_name"])
            theorems.append(thm)
        # split theorems into train and val
        num_train = int(len(theorems) * self.hparams.train_size)
        self.train_data = TheoremDataPipe(theorems[:num_train])
        self.val_data = TheoremDataPipe(theorems[num_train:])

    def train_dataloader(self):
        # data loader init args are copied from original code base
        # - batch_size=None (no batching), num_workers=0 (no new threads)
        return DataLoader(self.train_data, shuffle=True, batch_size=None, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=None, num_workers=0)


class TheoremDataPipe(MapDataPipe):
    # def __init__(self, theorems) -> None:
    def __init__(self, theorems) -> None:
        super().__init__()
        self.theorems = theorems 

    def __len__(self):
        return len(self.theorems)

    def __getitem__(self, index):
        return self.theorems[index]
