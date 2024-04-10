import argparse
import json
import random
import torch
from datasets import Dataset, load_dataset 
from itertools import islice
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    # BitsAndBytesConfig,
)
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from utils import add_pad_token, prepend_repo_root
from typing import Optional


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
    model,
    tokenizer,
    prompt_completion_pairs,
    sep="",
):
    input_texts = []
    # idx of first completion char + last completion char
    start_char_idxs = []
    end_char_idxs = []
    for prompt, completion in prompt_completion_pairs:
        text = prompt + sep + completion
        input_texts.append(text)
        start_char_idxs.append(len(prompt) + len(sep))
        end_char_idxs.append(len(text) - 1)
    
    batch_enc = tokenizer(input_texts, padding=True, return_tensors="pt")
    input_ids = batch_enc.input_ids
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    start_token_idxs = [batch_enc.char_to_token(i, char_idx) for i, char_idx in enumerate(start_char_idxs)]
    end_token_idxs = [batch_enc.char_to_token(i, char_idx) for i, char_idx in enumerate(end_char_idxs)]

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_probs, start_idx, end_idx in zip(gen_probs, start_token_idxs, end_token_idxs):
        completion_log_prob = input_probs[start_idx:end_idx+1].sum().item()
        batch.append({
            "log_prob_sum": completion_log_prob,
            "token_count": end_idx - start_idx + 1,
        })
    return batch

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
