import hydra
import os
from dotenv import load_dotenv

def prepare_environment_for_lean_dojo(relative_config_path: str = "../configs"):
    # github access token
    if not "GITHUB_ACCESS_TOKEN" in os.environ:
        with hydra.initialize(config_path=relative_config_path, version_base=None):
            config = hydra.compose(config_name="train")
        load_dotenv(config.env_paths.github_access_token)

    # lean dojo cache path
    cache_path_key = "CACHE_DIR"
    if (
        not cache_path_key in os.environ 
        and config.env_paths.lean_dojo_cache_path is not None
    ):
        os.environ[cache_path_key] = config.env_paths.lean_dojo_cache_path