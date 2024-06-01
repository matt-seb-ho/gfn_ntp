import hydra
import os
from typing import Optional
from dotenv import load_dotenv
from omegaconf import OmegaConf
from functools import lru_cache


HF_ACCESS_TOKEN_VAR_NAME = "HF_ACCESS_TOKEN"


def make_path_relative_to_repo(relative_path: str) -> str:
    # pre-condition: this function is defined in `repo/src/utils.py`
    return os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", relative_path)
    )


def load_github_access_token():
    if "GITHUB_ACCESS_TOKEN" in os.environ:
        return
    with hydra.initialize(config_path="../configs", version_base=None):
        config = hydra.compose(config_name="train")
    load_dotenv(config.env_files.github_access_token)


# cache individual attribute imports
@lru_cache(maxsize=None)
def import_attr_from_lean_dojo(attr_name):
    load_github_access_token()
    import lean_dojo
    return getattr(lean_dojo, attr_name)


# wrapper for multiple attributes
def import_from_lean_dojo(*args):
    if len(args) == 0:
        load_github_access_token()
        import lean_dojo
        return lean_dojo
    if len(args) == 1:
        return import_attr_from_lean_dojo(args[0])
    return tuple(import_attr_from_lean_dojo(arg) for arg in args)


def get_hf_access_token(
    dotenv_path: Optional[str] = None, 
    relative_to_repo: bool = False
) -> str:
    if HF_ACCESS_TOKEN_VAR_NAME in os.environ:
        return os.environ[HF_ACCESS_TOKEN_VAR_NAME]
    if dotenv_path is None:
        dotenv_path = make_path_relative_to_repo(".env")
    elif relative_to_repo:
        dotenv_path = make_path_relative_to_repo(dotenv_path)
    load_dotenv(dotenv_path)
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