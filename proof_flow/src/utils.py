import os
from functools import cache, lru_cache
from pathlib import Path
from typing import Optional

import hydra
from dotenv import load_dotenv
from omegaconf import OmegaConf

HF_ACCESS_TOKEN_VAR_NAME = "HF_ACCESS_TOKEN"


@cache
def get_config(config_path: str = "../../configs", config_name: str = "train") -> OmegaConf:
    with hydra.initialize(config_path=config_path, version_base=None):
        config = hydra.compose(config_name=config_name)
    return config


@cache
def repo_root() -> Path:
    config = get_config()
    return Path(config.paths.repo)


def prepare_environment_for_lean_dojo():
    # github access token
    if not "GITHUB_ACCESS_TOKEN" in os.environ:
        load_dotenv(get_config().paths.github_access_token)

    # lean dojo cache path
    cache_path_key = "CACHE_DIR"
    if not cache_path_key in os.environ:
        os.environ[cache_path_key] = get_config().paths.lean_dojo_cache_path


# cache individual attribute imports
@lru_cache(maxsize=None)
def import_attr_from_lean_dojo(attr_name):
    prepare_environment_for_lean_dojo()
    import lean_dojo
    return getattr(lean_dojo, attr_name)


# wrapper for multiple attributes
def import_from_lean_dojo(*args):
    if len(args) == 0:
        prepare_environment_for_lean_dojo()
        import lean_dojo
        return lean_dojo
    if len(args) == 1:
        return import_attr_from_lean_dojo(args[0])
    return tuple(import_attr_from_lean_dojo(arg) for arg in args)


def get_hf_access_token(): 
    config = get_config()
    load_dotenv(config.paths.hf_access_token)
    return os.getenv(HF_ACCESS_TOKEN_VAR_NAME)


def add_pad_token(model, tokenizer, pad_token="<pad>", padding_side="right"):
    # CodeLlama/Llama2 does not have a default mask/pad token
    # https://github.com/TrelisResearch/llama-2-setup
    # Check if the pad token is already in the tokenizer vocabulary
    if '<pad>' not in tokenizer.get_vocab():
        # Add the pad token
        tokenizer.add_special_tokens({"pad_token": pad_token})

    # resize the embeddings
    model.resize_token_embeddings(len(tokenizer))

    # configure the pad token in the model
    model.config.pad_token_id = tokenizer.pad_token_id

    # check if they are equal
    assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

    # print the pad token ids
    print('Tokenizer pad token ID:', tokenizer.pad_token_id)
    print('Model pad token ID:', model.config.pad_token_id)
    print('Model config pad token ID:', model.config.pad_token_id)
    
    # to prevent warnings
    tokenizer.padding_side = padding_side
