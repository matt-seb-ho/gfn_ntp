import hydra
import os
from dotenv import load_dotenv

def load_github_access_token():
    if "GITHUB_ACCESS_TOKEN" in os.environ:
        return
    with hydra.initialize(config_path="../configs", version_base=None):
        config = hydra.compose(config_name="train")
    load_dotenv(config.env_files.github_access_token)
