import argparse
import json
import torch
from datasets import Dataset, load_dataset 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm import tqdm
from utils import prepend_repo_root

DPO_DATA_PATH = prepend_repo_root("data/paired_random_train.json")
# PAIR_DATA_KEYS = ("prompt", "chosen", "rejected")
PAIR_DATA_KEYS = ("state", "positive", "negative")
PROMPT_KEY, CHOSEN_KEY, REJECTED_KEY = PAIR_DATA_KEYS
OUTPUT_FILE = prepend_repo_root("outputs/verify_eval.json")
MODEL_ID = "EleutherAI/llemma_7b"

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



def evaluate_verifier(model, tokenizer, pair_data, output_file=None):
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
    with open(DPO_DATA_PATH) as f:
        pref_data = json.load(f)
    canary = pref_data[0]
    assert set(PAIR_DATA_KEYS) == set(canary.keys()), "bad keys"

    # psr = argparse.ArgumentParser()
    # psr.add_argument("--out")
    # args = psr.parse_args()

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    acc, details = evaluate_verifier(model, tokenizer, pref_data, output_file=OUTPUT_FILE)


if __name__ == "__main__":
    main()