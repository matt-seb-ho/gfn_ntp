import argparse
import gc
import json
import random
from itertools import islice
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM  # BitsAndBytesConfig,
from transformers import AutoTokenizer


# Sequence probability
# - https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
def token_probabilities(model, tokenizer, input_texts):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch


def batch_completion_probabilities(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_completion_pairs: list[tuple[str, str]],
    sep: str = "",
    device: Optional[str | torch.device] = None,
    split_and_retry: bool = True,
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
    try:
        outputs = model(**batch_enc) # ensure attention mask is passed
    except RuntimeError as e:
        if (
            not split_and_retry 
            or "out of memory" not in str(e)
            or len(prompt_completion_pairs) == 1
        ):
            raise e
        # split the batch in half and retry
        torch.cuda.empty_cache()
        gc.collect()
        halfway_idx = len(prompt_completion_pairs) // 2
        half1 = prompt_completion_pairs[:halfway_idx]
        half2 = prompt_completion_pairs[halfway_idx:]
        sub_results = [
            batch_completion_probabilities(
                model, 
                tokenizer, 
                half_pairs,
                sep=sep, 
                device=device, 
                split_and_retry=True,
            )
            for half_pairs in (half1, half2)
        ]
        return sub_results[0] + sub_results[1]

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


def batch_sequence_probabilities(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sequences: list[str],
    termination_token_id: Optional[int] = None,
):
    batch_enc = tokenizer(sequences, padding=True, return_tensors="pt")
    input_ids = batch_enc.input_ids
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    
    # need to figure out where to stop summing probabilities
    # - this happens at the termination token
    termination_token_id = termination_token_id or tokenizer.eos_token_id
    gen_lengths = (input_ids == termination_token_id).byte().argmax(dim=-1).unsqueeze(-1)
    seq_probs = torch.gather(gen_probs.cumsum(dim=-1), -1, gen_lengths).squeeze(-1)
    return seq_probs


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
