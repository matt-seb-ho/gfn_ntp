import torch
import os
import sys
from dataclasses import dataclass
from functools import cache
from itertools import islice
from pathlib import Path
from typing import Optional

import hydra
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


HF_ACCESS_TOKEN_VAR_NAME = "HF_ACCESS_TOKEN"
DEFAULT_PAD_TOKEN = "<pad>"
# example of special tokens:
# - llemma: <s>, </s>
# - deepseek: <｜begin▁of▁sentence｜>, <｜end▁of▁sentence｜>
# - internlm math: <s>, </s>
CUSTOM_LOG_LEVEL = "GFN_DEBUG"


@dataclass
class SearchEvalConfig:
    step_interval: int = 250
    num_sampled_tactics: int = 8
    timeout: int = 30
    max_expansions: Optional[int] = None
    num_workers: int = 1
    num_gpus: int = 1
    max_input_seq_len: int = 130
    max_new_tokens: int = 30
    length_penalty: float = 0.0
        

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
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # check if they are equal
    assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

    # print the pad token ids
    print("Tokenizer pad token ID:", tokenizer.pad_token_id)
    print("Model pad token ID:", model.config.pad_token_id)
    
    # this determines whether padding tokens are added to the left or right side of the input
    # the common rule of thumb is that for auto-regressive models, padding should be on the left
    tokenizer.padding_side = padding_side



# reference:
# https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
def batch_completion_probabilities(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_completion_pairs: list[tuple[str, str]],
    sep: str = "",
    device: Optional[str | torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_texts = []
    start_char_idxs = [] # index of the first completion token
    end_char_idxs = []   # index of the last completion token
    for prompt, completion in prompt_completion_pairs:
        text = prompt + sep + completion
        input_texts.append(text)
        start_char_idxs.append(len(prompt) + len(sep))
        end_char_idxs.append(len(text) - 1)
    
    batch_enc = tokenizer(input_texts, padding=True, return_tensors="pt")
    if device:
        batch_enc = batch_enc.to(device)
    input_ids = batch_enc.input_ids
    outputs = model(**batch_enc) # ensure attention mask is passed
    log_prob_distributions = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token
    # - softmax(logits[b, i, :]) is the probability distribution for the token AFTER input_ids[b, i]
    # - (consequence 1): we don't care about the distribution after the last token in the input_ids
    log_prob_distributions = log_prob_distributions[:, :-1, :]
    # - while we have the distribution over the vocab after every token i,
    #   we only actually care about the probability of the actual next token
    # - i.e., we want softmax(logits[b, i, :])[input_ids[b, i+1]]
    # - notice the idx used for input_ids is i+1 while the idx used for log_probs is i
    # - for the gather operation, we need them aligned
    # - (consequence 2): we shift the input_ids forwards/leftwards by 1
    input_ids = input_ids[:, 1:]
    # - while logits has shape (batch_size, seq_len, vocab_size),
    #   and prob_distributions has shape (batch_size, seq_len - 1, vocab_size),
    #   the gather result has shape (batch_size, seq_len - 1)
    # - it's a batch of log probabilities of the actual sequence's tokens
    seq_log_probs = torch.gather(log_prob_distributions, 2, input_ids[:, :, None]).squeeze(-1)

    # version 1
    # completion_log_probs = []
    # for i in range(len(input_texts)):
    #     # start: token index of the first completion token
    #     # - the -1 is because the input_ids were shifted forward by 1
    #     # stop: token index of the token after the last completion token
    #     # - we use it as the end slice index, so we don't need to do -1
    #     start = batch_enc.char_to_token(i, start_char_idxs[i]) - 1
    #     stop = batch_enc.char_to_token(i, end_char_idxs[i])
    #     completion_log_prob = seq_log_probs[i, start:stop].sum().item()
    #     completion_log_probs.append({
    #         "log_prob_sum": completion_log_prob,
    #         "token_count": stop - start,
    #     })
    # return completion_log_probs

    # version 2
    start = []
    stop = []
    for i in range(len(input_texts)):
        # start: token index of the first completion token
        # - the -1 is because the input_ids were shifted forward by 1
        # stop: token index of the token after the last completion token
        # - we use it as the end slice index, so we don't need to do -1
        # NOTE: char_to_token returns None if char_idx is out of bounds
        # - this only happens when completion is empty
        start.append(batch_enc.char_to_token(i, start_char_idxs[i]) - 1)
        stop.append(batch_enc.char_to_token(i, end_char_idxs[i]))
    start = torch.tensor(start, device=device).unsqueeze(1)
    stop = torch.tensor(stop, device=device).unsqueeze(1)
    idx = (
        torch.arange(seq_log_probs.shape[1], device=seq_log_probs.device)
        .unsqueeze(0)
        .expand(seq_log_probs.shape[0], -1)
    )
    mask = (idx >= start) & (idx < stop)
    completion_log_probs = (seq_log_probs * mask).sum(dim=1)
    return completion_log_probs, (stop - start).squeeze(-1)


def batch_iterator(iterable, batch_size):
    """
    Generator that yields fixed-size batches from the input iterable.
    
    Parameters:
    - iterable: An iterable from which to yield batches.
    - batch_size: The size of each batch as an integer.
    
    Yields:
    - Batches of the input iterable, each batch is of size batch_size.
    """
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def batch_iterator_zip(iterables, batch_size):
    """
    Generator that yields fixed-size batches from the input iterables.
    
    Parameters:
    - iterables: A list of iterables from which to yield batches.
    - batch_size: The size of each batch as an integer.
    
    Yields:
    - Batches of the input iterables, each batch is of size batch_size.
    - If batch_size = n and we had m iterables, batches have shape 
    [[iter1_item1, ..., iter1_itemn], ..., [iterm_item1, ..., iterm_itemn]]

    Note:
    - if any of the iterables is exhausted, the generator stops.
    """
    iterators = [iter(it) for it in iterables]
    while True:
        batch = [list(islice(it, batch_size)) for it in iterators]
        if not all(batch):
            break
        yield batch


def disable_tokenizer_parallelism():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_up_debug_logging(cfg: OmegaConf):
    if cfg.include_lean_dojo_debug:
        level = "DEBUG"
    else:
        # between DEBUG (10) and INFO (20)
        logger.level(CUSTOM_LOG_LEVEL, no=15)
    if cfg.log_debug_to_stdout:
        logger.add(sys.stdout, level=level)
    if cfg.write_to_file:
        logger.add(repo_root() / cfg.debug_log_file, level=level)
    return level
