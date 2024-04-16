import hashlib
import json
import os
import warnings
from typing import Optional

from lean_dojo import LeanGitRepo, Pos, Theorem, is_available_in_cache
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe

warnings.filterwarnings("ignore", ".*does not have many workers.*")


class NTPDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        train_size=0.95,
        limit_theorems=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_data = None
        self.val_data = None

    def setup(self, stage):
        _, theorems, _ = _get_theorems(
            self.hparams.data_path,
            stage,
            num_theorems=self.hparams.limit_theorems,
        )
        num_train = int(len(theorems) * self.hparams.train_size)
        self.train_data = TheoremDataPipe(theorems[:num_train])
        self.val_data = TheoremDataPipe(theorems[num_train:])

    def train_dataloader(self):
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

# _get_theorems and _get_theorems_from_files are slightly modified from:
# https://github.com/lean-dojo/ReProver/blob/main/prover/evaluate.py
UNTRACED_MSG = "{r} has not been traced yet. Please use LeanDojo to trace it so that it's available in the cache."
def _get_theorems(
    data_path: str,
    split: str,
    file_path: Optional[str] = None,
    full_name: Optional[str] = None,
    name_filter: Optional[str] = None,
    num_theorems: Optional[int] = None,
) -> tuple[LeanGitRepo, list[Theorem], list[Pos]]:
    repo, theorems, positions = _get_theorems_from_files(
        data_path,
        split,
        file_path,
        full_name,
        name_filter,
        num_theorems,
    )
    all_repos = {thm.repo for thm in theorems}
    for r in all_repos:
        assert is_available_in_cache(r), UNTRACED_MSG.format(r=r)
    return repo, theorems, positions

def _get_theorems_from_files(
    data_path: str,
    split: str,
    file_path: Optional[str] = None,
    full_name: Optional[str] = None,
    name_filter: Optional[str] = None,
    num_theorems: Optional[int] = None,
) -> tuple[LeanGitRepo, list[Theorem], list[Pos]]:
    data = json.load(open(os.path.join(data_path, f"{split}.json")))
    theorems = []
    positions = []

    for t in data:
        skip = (
            file_path is not None and t["file_path"] != file_path
            or full_name is not None and t["full_name"] != full_name
            or (
                name_filter is not None 
                and not hashlib.md5(t["full_name"].encode()).hexdigest().startswith(name_filter)
            )
        )
        if skip:
            continue
        repo = LeanGitRepo(t["url"], t["commit"])
        theorems.append(Theorem(repo, t["file_path"], t["full_name"]))
        positions.append(Pos(*t["start"]))
    # jointly sort theorems and positions
    theorems_and_positions = list(zip(theorems, positions))
    theorems_and_positions.sort(key=lambda x: hashlib.md5(f"{x[0].file_path}:{x[0].full_name}".encode()).hexdigest())
    theorems, positions = zip(*theorems_and_positions)
    theorems, positions = list(theorems), list(positions)
    if num_theorems is not None:
        theorems = theorems[:num_theorems]
        positions = positions[:num_theorems]

    metadata = json.load(open(os.path.join(data_path, "../metadata.json")))
    repo = LeanGitRepo(metadata["from_repo"]["url"], metadata["from_repo"]["commit"])

    return repo, theorems, positions

"""
Usage
- data_path: to lean_dojo_benchmark_4/random/
- split: test
- file_path: None
- full_name: None
- name_filter: None
- num_theorems: 64
###
    repo, theorems, positions = _get_theorems(
        data_path, split, file_path, full_name, name_filter, num_theorems
    )
### 
    pass_1, trees = sample_trees(
        args.data_path,
        args.exp_id,
        args.split,
        args.file_path,
        args.full_name,
        args.name_filter,
        args.num_theorems,
        args.ckpt_path,
        args.indexed_corpus_path,
        args.tactic,
        args.module,
        args.num_sampled_tactics,
        args.timeout,
        args.num_cpus,
        args.with_gpus,
        args.verbose,
        hf_generator_id=args.hf_gen_id,
        hf_retrieval_id=args.hf_ret_id,
        output_tree_file=args.output_tree_file
    )
###
CUDA_VISIBLE_DEVICES=0,1,2 python -m prover.sample \
  --data-path data/leandojo_benchmark_4/random/ \
  --hf_gen_id kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small \
  --hf_ret_id kaiyuy/leandojo-lean4-retriever-byt5-small \
  --indexed-corpus-path outputs/indexed_corpus.pickle \
  --split test \
  --num-cpus 3 \
  --with-gpus \
  --num-theorems 64 \
  --output_dir outputs/pickle_jar2 \
  --lean_dojo_cache_path /mnt/hdd/msho/gfn_ntp/.cache/lean_dojo
"""
