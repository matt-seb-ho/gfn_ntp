import argparse
import json
import random
from itertools import islice

import torch
from datasets import Dataset, load_dataset
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,  # BitsAndBytesConfig,
                          AutoTokenizer)
from utils import add_pad_token, prepend_repo_root

DPO_EVAL_DATA_PATH = prepend_repo_root("data/paired_random_val.json")
# PAIR_DATA_KEYS = ("prompt", "chosen", "rejected")
RANDOM_SUBSET = True
RANDOM_SEED = 42
SUBSET_SIZE = 100
BATCH_SIZE = 1
PAIR_DATA_KEYS = ("state", "positive", "negative")
PROMPT_KEY, CHOSEN_KEY, REJECTED_KEY = PAIR_DATA_KEYS
OUTPUT_FILE = prepend_repo_root("outputs/verify_eval_base100.json")
MODEL_ID = "EleutherAI/llemma_7b"

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

# old impl
def lm_log_prob(
    sequence, 
    model, 
    tokenizer, 
    exclude_end_token=True,
    debug=False
):
    inputs = tokenizer(sequence, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        if model.config.is_encoder_decoder:
            outputs = model(
                **inputs, 
                decoder_input_ids=model._shift_right(inputs["input_ids"])
            )
        else:
            outputs = model(**inputs)
    # next_token_distr = torch.log(softmax(outputs.logits.float()))
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

    token_length = log_probs.shape[1]
    if exclude_end_token:
        token_length -= 1

    log_prob = 0
    for enc_idx in range(token_length):
        target_id = inputs["input_ids"][0, enc_idx] 
        log_prob += log_probs[0, enc_idx, target_id].item()
        if debug:
            current = tokenizer.decode(inputs["input_ids"][0, enc_idx - 1])
            next_token = tokenizer.decode(target_id)
            lp = log_probs[0, enc_idx, target_id].item()
            print(f"(debug) log_prob from [{current}] to [{next_token}]: {lp}")
        
    return {
        "log_prob": log_prob,
        "token_length": token_length
    }

def lm_completion_log_prob(
    context, 
    completion, 
    model, 
    tokenizer, 
    exclude_end_token=True,
    normalize_by="length",
    debug=False
):
    if debug:
        print(f"complete passage: {context}{completion}")

    inputs = tokenizer(context + completion, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        # decoder-only: outputs = model(**inputs, labels=inputs["input_ids"])
        outputs = model(**inputs, decoder_input_ids=model._shift_right(inputs["input_ids"]))

    # more numerically stable using built-in composition
    # next_token_distr = torch.log(softmax(outputs.logits.float()))
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    
    """
    v1 (broken)
    in previous implementation, diff_len computation is bugged.
    >>> tokenizer.encode(s: str, return_tensors="pt") 
    returns a tensor with shape [1, num_tokens].
    len(.) the result yields 1
    # context_only = tokenizer.encode(context, return_tensors="pt")
    # diff_len = inputs["input_ids"].shape[1] - len(context_only)

    v2 (still broken)
    completion_token_length = (
        inputs["input_ids"].shape[1]      # total length
        - len(tokenizer.encode(context))  # minus context length
    
    the idea for v1/v2 was to iterate over 
    >>> range(-completion_token_length, 0 if include_end_token else - 1) 
    but this doesn't work because sometimes the completion is a single word
    which is tokenized to 1 word, while the whole sequence is tokenized to include 
    an eos token (</s> for flan-t5). So this method would only look at the </s> token probability.
    )

    # I think the fix is to use BatchEncoding.char_to_token to find the definitive starting point
    """
    
    completion_log_prob = 0

    # start corresponds to the *input* idx of the first completion token
    start = inputs.char_to_token(len(context))
    end = outputs.logits.shape[1]
    if exclude_end_token:
        end -= 1

    for enc_idx in range(start, end):
        # log_probs[0, idx] = vocab probability distribution AFTER seeing DECODER INPUT token_idx
        """
        outputs (logits) are right shifted in enc-dec architecture. As such:
        - target_id[0, idx] is the token id we're trying to get the log_prob of.
        - log_probs[0, idx, target_id] is the probability of target_id 
          after seeing decoder input idx (encoder input idx - 1)
        
        """

        target_id = inputs["input_ids"][0, enc_idx] 
        completion_log_prob += log_probs[0, enc_idx, target_id].item()
        if debug:
            current = tokenizer.decode(inputs["input_ids"][0, enc_idx - 1])
            next_token = tokenizer.decode(target_id)
            lp = log_probs[0, enc_idx, target_id].item()
            print(f"(debug) log_prob from [{current}] to [{next_token}]: {lp}")
        
    res = {
        "log_prob": completion_log_prob,
        "token_length": end - start
    }

    if normalize_by == "length":
        res["normalized"] = completion_log_prob / res["token_length"]
    elif normalize_by == "unconditional_prob":
        # unconditional_probability: normalize by the lm-probability of the completion on its own
        unconditional = lm_log_prob(completion, model, tokenizer, exclude_end_token)
        res["normalized"] = completion_log_prob - unconditional["log_prob"]
    elif normalize_by and debug:
        print(
            "Unknown normalized_by value. Valid options include:\n"
            "(1) length\n"
            "(2) unconditional_prob"
        )

    return res

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

def evaluate_verifier(
    model, 
    tokenizer, 
    pair_data, 
    output_file=None, 
    batch_size=1,
): 
    results = []
    correct = 0
    correct_norm = 0
    for batch in tqdm(batch_iterator(pair_data, batch_size), total=(len(pair_data) // batch_size) + 1):
        chosen_pairs = [(entry[PROMPT_KEY], entry[CHOSEN_KEY]) for entry in batch]
        rejected_pairs = [(entry[PROMPT_KEY], entry[REJECTED_KEY]) for entry in batch]
        chosen_scores = batch_completion_probabilities(
            model, tokenizer, chosen_pairs
        )
        reject_scores = batch_completion_probabilities(
            model, tokenizer, rejected_pairs 
        )
        for e, cs, rs in zip(batch, chosen_scores, reject_scores):
            c_lp_norm = cs["log_prob_sum"] / cs["token_count"]
            r_lp_norm = rs["log_prob_sum"] / rs["token_count"]
            result_entry = {
                "prompt": e[PROMPT_KEY],
                "chosen": e[CHOSEN_KEY],
                "reject": e[REJECTED_KEY],
                "c_log_prob": cs["log_prob_sum"],
                "c_token_length": cs["token_count"],
                "c_log_prob_norm": c_lp_norm,
                "r_log_prob": rs["log_prob_sum"],
                "r_token_length": rs["token_count"],
                "r_log_prob_norm": r_lp_norm,
                "correct": cs["log_prob_sum"] > rs["log_prob_sum"],
                "correct_norm": c_lp_norm > r_lp_norm,
            }
            correct += result_entry["correct"]
            correct_norm += result_entry["correct_norm"]
            results.append(result_entry)
    
    stats = {
        "correct": correct,
        "correct_norm": correct_norm,
        "total": len(pair_data),
        "prop_correct": correct / len(pair_data),
        "prop_correct_norm": correct_norm / len(pair_data),
    }
    print(json.dumps(stats, indent=2))
    
    if output_file:
        with open(output_file, 'w') as f:
            obj_to_save = {"stats": stats, "results": results}
            json.dump(obj_to_save, f)

    return stats, results
        

# old impl
def _evaluate_verifier(
    model, 
    tokenizer, 
    pair_data, 
    output_file=None, 
    # batch_size=1,
):
    correct = 0
    scores = []
    for i, entry in enumerate(tqdm(pair_data, total=len(pair_data))):
        prompt = entry[PROMPT_KEY]
        # chosen_score = score_completion(model, tokenizer, prompt, entry[CHOSEN_KEY])
        # reject_score = score_completion(model, tokenizer, prompt, entry[REJECTED_KEY])
        chosen_score = lm_completion_log_prob(prompt, entry[CHOSEN_KEY], model, tokenizer)
        reject_score = lm_completion_log_prob(prompt, entry[REJECTED_KEY], model, tokenizer)
        scores.append((chosen_score, reject_score))
        if chosen_score > reject_score:
            correct += 1
    
    acc = correct / len(pair_data)
    details = {
        "correct": correct,
        "total": len(pair_data),
        "acc": acc,
        "scores": scores
    }
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output_file, f, indent=2)
    return acc, details
    

def main():
    psr = argparse.ArgumentParser()
    psr.add_argument("--hf_model_id", type=str, required=True)
    psr.add_argument("--output_file", type=str, required=True)
    psr.add_argument("--input_data", type=str, default=DPO_EVAL_DATA_PATH)
    psr.add_argument("--rng_seed", type=int, default=42)
    psr.add_argument("--subset_size", type=int, default=100)
    psr.add_argument("--batch_size", type=int, default=1)
    psr.add_argument("--peft", action="store_true")
    psr.add_argument("--quantize", action="store_true")
    psr.add_argument("--flash_attn", action="store_true")
    args = psr.parse_args()
    
    # random.seed(RANDOM_SEED)
    random.seed(args.rng_seed)
    with open(args.input_data) as f:
        pref_data = json.load(f)["pairs"]
    # if RANDOM_SUBSET:
    if args.subset_size:
        pref_data = random.choices(pref_data, k=args.subset_size)

    canary = pref_data[0]
    assert set(PAIR_DATA_KEYS) == set(canary.keys()), "bad keys"

    # psr = argparse.ArgumentParser()
    # psr.add_argument("--out")
    # args = psr.parse_args()

    model_init_kwargs = {}
    if args.quantize:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True, 
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config
        model_init_kwargs["torch_dtype"] = torch.bfloat16
        model_init_kwargs["quantization_config"] = bnb_config
    
    if args.flash_attn:
        # attn_implementation="flash_attention_2", # maybe comment this line
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
        
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id)
    model_cls = AutoPeftModelForCausalLM if args.peft else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        args.hf_model_id,
        device_map="auto",
        # bells and whistles
        **model_init_kwargs
        # attn_implementation="flash_attention_2", # maybe comment this line
        # torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config
    )
    add_pad_token(model, tokenizer)
    stats, details = evaluate_verifier(
        model, 
        tokenizer, 
        pref_data, 
        output_file=args.output_file,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()