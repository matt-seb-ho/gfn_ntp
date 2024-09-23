import hydra
from omegaconf import DictConfig
from proof_flow.scripts.gfn_tuning.train import train_setup

# relative to this file (proof_flow/scripts/gfn_tuning/single_thm.py)
CONFIG_DIR = "../../../configs/"

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train_five")
def train_five_theorems(config: DictConfig):
    task, data, trainer = train_setup(config)
    trainer.fit(model=task, datamodule=data)

if __name__ == "__main__":
    train_five_theorems()
