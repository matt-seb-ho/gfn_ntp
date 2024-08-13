import gzip
import heapq
import pickle
from collections import defaultdict
from typing import Optional

import editdistance
import numpy as np
import spacy
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from proof_flow.src.constants import (
    DEFAULT_VERIFIER_ADAPTER_NAME,
    DEFAULT_VERIFIER_BATCH_SIZE,
    PROOF_COMPLETE_MESSAGE
)
from proof_flow.src.gfn_tuning.verifier import (
    batch_completion_probabilities,
    batch_iterator
)


def lora_to_base(model):
    model.base_model.disable_adapter_layers()
    model.eval()

def base_to_lora(model):
    model.base_model.enable_adapter_layers()
    model.train()

def compute_log_reward(
    states: list[list[str]],
    tactics: list[list[str]],
    model: PeftModel,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Computes reward for a batch of trajectores (states, tactics) using heuristics and model.
    
    Arguments
        states: 2D list of shape (batch_size, trajectory_length) containing states
        tactics: 2D list of shape (batch_size, trajectory_length - 1) containing tactics
        model: verifier for scoring the generated text
        tokenizer: AutoTokenizer for encoding the input
    """
    assert len(states) == len(tactics)
    # r = max(is_correct - 1, model_score)
    # - heuristic score
    is_correct = torch.tensor([[1 if state[-1] == PROOF_COMPLETE_MESSAGE else 0 for state in states]])
    
    # - model based score
    prompt_completion_pairs = []
    batch_idx_to_pair_idx = defaultdict(list)
    for i in len(states):
        for state, tactic in zip(states[i], tactics[i]):
            batch_idx_to_pair_idx[i].append(len(prompt_completion_pairs))
            prompt_completion_pairs.append((state, tactic))
    results = []
    for batch in batch_iterator(prompt_completion_pairs, batch_size):
        results.extend(batch_completion_probabilities(model, tokenizer, batch))
    model_scores = torch.tensor([
        sum(results[j]["log_prob_sum"] for j in batch_idx_to_pair_idx[i])
        for i in range(len(states))
    ])

    return torch.log(torch.max(torch.stack(is_correct - 1, model_scores, dim=-1), dim=-1))
        
def format_prompt(initial_state: str, tactic: Optional[str], resulting_state: Optional[str]) -> str:
    # TODO: finalize this to match the verifier's training
    if tactic is not None:
        if resulting_state is None:
            return f"{initial_state}\n###\n{tactic}\n###\n{resulting_state}"
        return f"{initial_state}\n###\n{tactic}"
    return f"{initial_state}\n###\n"


class NTPReward:
    def __init__(
        self, 
        model: Optional[PeftModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        model_id: Optional[str] = None, 
        temperature: float = 1.0, 
        verifier_batch_size: Optional[int] = None,
        model_loading_kwargs: Optional[dict] = None,
        verifier_adapter_name: str = DEFAULT_VERIFIER_ADAPTER_NAME,
    ):
        self.temperature = temperature
        self.verifier_batch_size = verifier_batch_size
        self.verifier_adapter_name = verifier_adapter_name

        assert (model is None) == (tokenizer is None)
        assert model is not None or model_id is not None
        if model:
            self.model = model,
            self.tokenizer = tokenizer
        else:
            model_loading_kwargs = model_loading_kwargs or {}
            self.model = AutoModel.from_pretrained(model_id, **model_loading_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            # loads but does not activate the adapter
            model.load_adapter(verifier_adapter_name)
    
    def score(
        self,
        states: list[list[str]],
        tactics: list[list[str]],
        batch_size: Optional[int] = None
    ):
        batch_size = batch_size or self.verifier_batch_size or DEFAULT_VERIFIER_BATCH_SIZE
        # swap to verification adapters
        training = self.model.training
        previous_adapter = self.model.active_adapter
        self.model.set_adapter(self.verifier_adapter_name)
        self.model.eval()
        with torch.no_grad():
            log_reward = compute_log_reward(
                states, 
                tactics, 
                self.model, 
                self.tokenizer, 
                batch_size=batch_size
            )
        # reset model to original state
        self.model.set_adapter(previous_adapter)
        if training:
            self.model.train()
        return log_reward


@torch.no_grad()
def score_fast(
    model,
    encoded_input,
    termination_token_id,
    min_len,
    skip_first,
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    vocab_alpha=-99,
    prompt_cache=None,
):
    """
    Function for scoring (prompt/state, generated_text/tactic)
    """
    # caching for efficiency
    if prompt_cache is None:
        logits = model(encoded_input).logits
    else:
        # prompt_cache[1] contains past_key_values which need to be reshaped to the right batch size from encoded_input
        batched_prompt_cache = tuple(
            tuple(
                [
                    prompt_cache[1][i][j].repeat(encoded_input.shape[0], 1, 1, 1)
                    for j in range(len(prompt_cache[1][i]))
                ]
            )
            for i in range(len(prompt_cache[1]))
        )
        logits = model(encoded_input, past_key_values=batched_prompt_cache).logits

    # score the log probability of the input sequence while ignoring termination and padding tokens
    # - get rid of the first few tokens
    logits = logits[:, skip_first - 1 :]
    # - penalize non-'nice' and 'naughty' tokens
    if vocab_nice_mask is not None:
        logits[:, :, ~vocab_nice_mask] += vocab_alpha
    elif vocab_naughty_mask is not None:
        logits[:, :, vocab_naughty_mask] += vocab_alpha
    # - get the log probabilities
    logprob = logits.log_softmax(-1)
    token_ids = encoded_input[:, skip_first:].unsqueeze(-1)
    logPF = logprob[:, :-1].gather(-1, token_ids).squeeze(-1)
    logP = logPF.cumsum(dim=-1)  # logP(generated[:i+1] | prompt)
    # **Design Decision** how to compute p(terminate|incomplete_proof)?
    # - lazy choice: omit this term for first try
    # - other choice: some combination of p(eos|.), p('\n\n'|.), etc.
    # reward = logprob[
    #     :, :, termination_token_id
    # ]  # logP(generated[i+1]=term | prompt + generated[:i+1])
    # reward[:, 1:] += logP  # logP(generated[:i] + term | prompt)
    reward = logP
    non_term_mask = (encoded_input != termination_token_id)[:, skip_first:]
    non_term_mask = torch.cat(
        (
            non_term_mask.new_ones(non_term_mask.shape[0], 1),
            non_term_mask,
        ),
        dim=-1,
    )  # Start (i.e., empty) state has never terminated
    reward[~non_term_mask] = 0.0
    reward_unpenalized = reward.clone()
    reward = torch.where(non_term_mask.cumsum(dim=-1) - 1 < min_len, -99, reward)
    return reward, reward_unpenalized


class FrozenModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_token_id,
        temperature=1.0,
        min_len=1,
        vocab_alpha=-50.0,
        vocab_nice_mask=None,
        vocab_naughty_mask=None,
        sentence_validator=None,
        valid_sentence_alpha=None,
    ):
        assert (
            sentence_validator is None
            and valid_sentence_alpha is None
            or sentence_validator is not None
            and valid_sentence_alpha is not None
        )

        self.temperature = temperature
        self.sentence_token_id = sentence_token_id
        self.vocab_nice_mask = vocab_nice_mask
        self.vocab_naughty_mask = vocab_naughty_mask
        self.vocab_alpha = vocab_alpha
        self.min_len = min_len
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha

    def score(self, input_batch, prompt_length, model, tokenizer):
        lora_to_base(model)
        training = model.training
        model.eval()
        reward, reward_unpenalized = score_fast(
            model=model,
            encoded_input=input_batch,
            termination_token_id=self.sentence_token_id,
            skip_first=prompt_length,
            vocab_nice_mask=self.vocab_nice_mask,
            vocab_naughty_mask=self.vocab_naughty_mask,
            vocab_alpha=self.vocab_alpha,
            min_len=self.min_len,
        )
        reward /= self.temperature
        reward_unpenalized /= self.temperature
        base_to_lora(model)
        if training:
            model.train()

        if self.sentence_validator is not None:
            invalid = self.sentence_validator(input_batch[:, prompt_length:], tokenizer)
            invalid = invalid * self.valid_sentence_alpha
            reward = torch.min(reward, invalid)

        return reward, reward_unpenalized


class SentenceValidator:
    def __init__(self, sentence_token_id) -> None:
        self.sentence_token_id = sentence_token_id

    def __call__(self, sentences, tokenizer):
        pass


class RuleSentenceValidator(SentenceValidator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nlp = spacy.load("en_core_web_lg")

    def __call__(self, sentences, tokenizer):
        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            dtype=torch.bool,
            device=sentences.device,
        )
        invalid[:, 0] = True  # Empty sentence is never valid
        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                if sentences[i, j] == self.sentence_token_id:
                    break  # Only unterminated sentences get a reward
                sent = tokenizer.decode(sentences[i, : j + 1])
                sent = self.nlp(sent).sents
                tokens = []
                for s in sent:
                    for t in s:
                        tokens.append(t)
                if not (len(tokens) >= 2 and tokens[0].is_space and tokens[1].is_title):
                    invalid[i, j + 1] = True  # Must start with a space and capital
                    continue
                has_noun = 1
                has_verb = 1
                for token in tokens:
                    if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                        has_noun -= 1
                    elif token.pos_ in ["VERB", "AUX"]:
                        has_verb -= 1
                if has_noun > 0 or has_verb > 0:
                    invalid[i, j + 1] = True  # Must have a noun and a verb
        return invalid


class ModelSentenceValidator(SentenceValidator):
    def __init__(self, *args, model_name=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if model_name is None:
            model_name = "textattack/roberta-base-CoLA"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, device_map="auto"
        )

    @torch.no_grad()
    def __call__(self, sentences, tokenizer):
        sentences = sentences.to(self.model.device)
        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            dtype=torch.bool,
            device=self.model.device,
        )
        invalid[:, 0] = True  # Empty sentence is never valid
        done = torch.zeros(sentences.shape[0]).bool().to(self.model.device)
        for i in range(sentences.shape[1]):
            sent = sentences[:, : i + 1]
            done |= sent[:, -1] == self.sentence_token_id
            if done.all():
                break
            sent = self.tokenizer(
                tokenizer.batch_decode(sent),
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            invalid_probs = self.model(**sent).logits.softmax(dim=-1)[:, 0]
            invalid[~done, i + 1] = invalid_probs[~done] > 0.2
        return invalid


def modified_subtb_loss(
    log_pf,
    log_r,
    log_pterm,
    generated_text,
    termination_token_id,
    prompt_len,
    subtb_lambda=1.0,
):
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_text.shape[1] - prompt_len
    )
    assert (
        log_pf.shape[1] > 1
    )  # With modified-style losses, we need at least one transition before terminating

    delta = (
        log_r[:, :-1]
        + log_pf[:, :-1]
        + log_pterm[:, 1:]
        - log_r[:, 1:]
        - log_pterm[:, :-1]
    )
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

    # Get a mask for tokens after the termination token in the generated_text
    mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(-1) >= 1

    batch_loss = 0.0
    total_lambda = 0.0
    generated_len = generated_text.shape[1] - prompt_len
    for subtraj_len in range(1, generated_len):
        subtb_term = (
            delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]
        ) ** 2
        subtb_term[mask[:, subtraj_len - 1 :]] = 0
        batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
        total_lambda += (
            subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
        )
    batch_loss /= total_lambda

    return batch_loss


def get_termination_vals(
    generated_text,
    log_pf,
    log_pterm,
    log_r,
    log_r_unpenalized,
    termination_token_id,
    prompt_len,
):
    batch_idx = torch.arange(generated_text.size(0))
    gen_len = (
        (generated_text[:, prompt_len:] == termination_token_id).byte().argmax(dim=-1)
    )
    if log_pf is None and log_pterm is None:
        log_pfs = None
    else:
        log_pf = torch.cat([torch.zeros_like(log_pf[:, :1]), log_pf], dim=-1)[:, :-1]
        log_pfs = log_pf.cumsum(dim=-1) + log_pterm
        log_pfs = log_pfs[batch_idx, gen_len]
    log_r = log_r[batch_idx, gen_len]
    log_r_unpenalized = log_r_unpenalized[batch_idx, gen_len]
    return log_pfs, log_r, log_r_unpenalized, gen_len


class SequenceDiversity:
    def __init__(self, method, **kwargs):
        self.method = method
        if method is None:
            pass
        elif method == "sequence_embedding":
            model_name = kwargs.get(
                "model_name", "sentence-transformers/all-mpnet-base-v2"
            )
            self.model = SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unknown sequence diversity method: {method}")

    @torch.no_grad()
    def __call__(self, sequences):
        if self.method is None:
            return None
        elif self.method == "sequence_embedding":
            embeddings = self.model.encode(sequences, show_progress_bar=False)
            sim = cos_sim(embeddings, embeddings)
            indices = torch.triu_indices(len(sequences), len(sequences), offset=1)
            diversity = 1 - sim[indices[0], indices[1]].mean().item()
        else:
            raise ValueError(f"Unknown sequence diversity method: {self.method}")
        return diversity
