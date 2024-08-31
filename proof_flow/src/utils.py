import os
from functools import cache, lru_cache
from pathlib import Path
from typing import Optional

import hydra
from dotenv import load_dotenv
from omegaconf import OmegaConf

HF_ACCESS_TOKEN_VAR_NAME = "HF_ACCESS_TOKEN"
DEFAULT_PAD_TOKEN = "<pad>"
# example of special tokens:
# - llemma: <s>, </s>
# - deepseek: <｜begin▁of▁sentence｜>, <｜end▁of▁sentence｜>
# - internlm math: <s>, </s>


@cache
def get_config(
    config_path: str = "../../configs", 
    config_name: str = "train",
    overrides: Optional[str] = None,
) -> OmegaConf:
    if overrides:
        overrides = overrides.split(",")
    with hydra.initialize(config_path=config_path, version_base=None):
        config = hydra.compose(config_name=config_name, overrides=overrides)
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
        config_cache_path = get_config().paths.lean_dojo_cache_path
        if config_cache_path is not None:
            os.environ[cache_path_key] = config_cache_path


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


def set_up_padding(
    model, 
    tokenizer, 
    pad_token=DEFAULT_PAD_TOKEN,
    padding_side="left",
    token_embedding_multiple=None,
):
    """
    configures model and tokenizer for padding (needed for batch processing)
    
    based on the following scrip/repo
    https://github.com/TrelisResearch/llama-2-setup
    """

    # check if the pad token is already in the tokenizer vocabulary
    if tokenizer.pad_token is not None:
        return

    # add the pad token
    tokenizer.add_special_tokens({"pad_token": pad_token})

    # resize the embeddings
    # - only necessary if the new tokenizer size > model's vocab size
    #    - for deepseek-prover v1, the vocab size was set to 102400 while the tokenizer only hadd 100002 tokens
    # - for hardware performance reasons, it's useful for the embedding matrix to have certain dimension
    #   https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
    # - the doc suggests a100 wants multiple of 64 for FP16
    # - unfortunately it doesn't say anything about bfloat16
    # - TODO: empirically test the performance of the model with token_embedding_multiple=64 vs. None
    # vocab size issue described here:
    # https://github.com/huggingface/transformers/issues/22312#issuecomment-1684141492
    # https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
    if model.config.vocab_size < len(tokenizer):
        if token_embedding_multiple is None:
            model.resize_token_embeddings(len(tokenizer))
        else:
            model.resize_token_embeddings(
                len(tokenizer),
                pad_to_multiple_of=token_embedding_multiple,
            )

    # configure the pad token in the model
    model.config.pad_token_id = tokenizer.pad_token_id

    # check if they are equal
    assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

    # print the pad token ids
    print("Tokenizer pad token ID:", tokenizer.pad_token_id)
    print("Model pad token ID:", model.config.pad_token_id)
    
    # this determines whether padding tokens are added to the left or right side of the input
    # the common rule of thumb is that for auto-regressive models, padding should be on the left
    tokenizer.padding_side = padding_side
