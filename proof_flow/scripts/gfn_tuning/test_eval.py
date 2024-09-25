import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from types import MethodType
from proof_flow.src.utils import get_config
from proof_flow.src.search.proof_search import Status
from proof_flow.scripts.gfn_tuning.train import train_setup
from loguru import logger


# relative to this file (proof_flow/scripts/gfn_tuning/train.py)
CONFIG_DIR = "../../../configs/"


def main(config: DictConfig):
    task, data, trainer = train_setup(config)
    model = task.model
    val_probes = task.hparams.search_eval_probes

    # Fix a bug that arises when using 4-bit quantized models.
    # It's caused by different operations being on different devices,
    # so we'll just deactivate lightning's automatic device placement
    # and let huggingface handle the dynamic device placement
    if config.task.training.use_4bit:
        task.to = MethodType(lambda s, _: s, task)
        task.cuda = MethodType(lambda s: s, task)

    # trainer.fit(model=task, datamodule=data)
    model.eval()
    task.hparams.sanity_check_probes = len(val_probes)
    results = task.run_proof_search_eval()
    name_to_idx = {
        thm_dict["full_name"]: thm_idx
        for thm_idx, thm_dict in enumerate(val_probes)
    }
    for r in results:
        if r is None:
            print("Discarded")
            continue
        # r is a SearchResult object
        proved = (r.status == Status.PROVED)
        thm_dict = val_probes[name_to_idx[r.theorem.full_name]]
        proof_len = len(thm_dict["traced_tactics"])
        logger.info(f"proved: {proved}, thm: {thm_dict['full_name']}, proof_len: {proof_len}")
        

if __name__ == "__main__":
    cfg = get_config(config_name="train_five")
    main(cfg)
