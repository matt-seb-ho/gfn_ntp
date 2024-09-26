import ray
import openai
from loguru import logger
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from peft import (
    AutoPeftModelForCausalLM, 
    AutoPeftModelForSeq2SeqLM,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
)
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from proof_flow.src.search.common import (
    remove_marks, 
    zip_strict, 
    format_augmented_state,
    _HuggingFaceLM,
)
from proof_flow.src.search.retriever_model import PremiseRetriever
from proof_flow.src.utils import prepare_environment_for_lean_dojo


prepare_environment_for_lean_dojo()
from lean_dojo import Pos # isort: skip


class TacticGenerator(ABC):
    """A tactic generator takes a state and generates multiple tactic candidates."""

    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError


class GPT4TacticGenerator(TacticGenerator):
    def __init__(
        self,
        organization: str,
        api_key: str,
        model: str = "gpt-4",
        max_tokens: int = 1024,
        num_retries: int = 3,
        threshold: float = 0.9,
    ):
        super().__init__()
        openai.organization = organization
        openai.api_key = api_key
        self.model = model
        self.default_prompt = "You are an expert in theorem proving in Lean. We are trying to solve the Lean theorem 'THEOREM_FULL_NAME' from the mathlib file 'FILE_PATH'. The current tactic state is: 'TACTIC_STATE'. Suggest exactly NUM_SAMPLES unique tactics to progress in solving 'THEOREM_FULL_NAME', along with their confidence levels as a float between 0 and 1. Rank them in order of effectiveness. Present the tactics and their confidence levels as comma-separated tuples in this format: #(tactic_{1}, confidence_{1})#, #(tactic_{2}, confidence_{2})#, ..., #(tactic_{NUM_SAMPLES}, confidence_{NUM_SAMPLES})#."
        self.max_tokens = max_tokens
        self.num_retries = num_retries
        self.threshold = threshold

    def initialize(self) -> None:
        pass

    async def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        prompt = (
            self.default_prompt.replace("TACTIC_STATE", state)
            .replace("FILE_PATH", file_path)
            .replace("THEOREM_FULL_NAME", theorem_full_name)
            .replace("NUM_SAMPLES", str(int(num_samples / self.threshold)))
        )
        logger.info(prompt)

        for _ in range(self.num_retries):
            response = None
            # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=0,
                    max_tokens=self.max_tokens,
                    # stop="E:" #
                )
            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                logger.info(f"OpenAI API returned an API Error: {e}")
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                logger.info(f"Failed to connect to OpenAI API: {e}")
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                logger.info(f"OpenAI API request exceeded rate limit: {e}")
                continue
            except Exception as e:
                logger.info(e)
                continue

            if response is None:
                continue

            logger.info(f"GPT-4 response: {response}")
            output = response["choices"][0]["message"]["content"]
            indices = []

            for i, c in enumerate(output):
                if c == "#":
                    indices.append(i)

            tactics_with_scores = []

            for i in range(1, len(indices), 2):
                tactic_and_confidence = output[indices[i - 1] + 1 : indices[i]].strip()

                try:
                    while tactic_and_confidence[0] == "(":
                        tactic_and_confidence = tactic_and_confidence[1:]

                    if tactic_and_confidence[-1] == ")":
                        tactic_and_confidence = tactic_and_confidence[:-1]

                    split_index = tactic_and_confidence.rindex(",")
                    tactic = tactic_and_confidence[:split_index].strip()
                    confidence = float(tactic_and_confidence[split_index + 1 :].strip())
                except Exception as e:
                    logger.info(e)
                    logger.info(
                        f"{self.model} output {output[indices[i-1]+1:indices[i]]} was not formatted correctly and could not be parsed."
                    )
                    continue

                tactics_with_scores.append((tactic, confidence))

            if len(tactics_with_scores) < int(self.threshold * num_samples):
                continue

            tactics_with_scores = sorted(
                tactics_with_scores, key=lambda x: x[1], reverse=True
            )[: min(num_samples, len(tactics_with_scores))]
            logger.debug(f"GPT-4 tactics: {tactics_with_scores}")
            logger.debug(
                f"GPT-4 tactic count requested: {num_samples} / {self.threshold} = {int(num_samples / self.threshold)}"
            )
            logger.debug(
                f"GPT-4 tactic count received and parsed: {len(tactics_with_scores)}"
            )
            return tactics_with_scores

        raise ValueError("GPT-4 outputs are unparsable.")


class FixedTacticGenerator(TacticGenerator):
    def __init__(self, tactic, module) -> None:
        self.tactic = tactic
        self.module = module

    def initialize(self) -> None:
        pass

    async def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        return [(f"{{ {self.tactic} }}", 1.0)]


class HuggingFaceGenerator(TacticGenerator):
    def __init__(
        self,
        model_path: str,
        device,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        max_new_tokens: int,
        length_penalty: float,
        template: str = "%s",
        is_peft_model: bool = False,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        use_beam_search: bool = True,
        model: Optional[_HuggingFaceLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        is_decoder_only: Optional[bool] = None,
        end_of_step_token_id: Optional[int] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.max_new_tokens = max_new_tokens
        self.length_penalty = length_penalty
        self.template = template
        self.is_peft_model = is_peft_model
        self.quantization_config = quantization_config
        self.use_beam_search = use_beam_search
        self.model = model
        self.tokenizer = tokenizer
        self.decoder_only = is_decoder_only
        self.end_of_step_token_id = (
            end_of_step_token_id
            or tokenizer.eos_token_id
        )


    def initialize(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            self.generator = self.model
            self.tokenizer = self.tokenizer
            if self.decoder_only is None:
                self.decoder_only = isinstance(
                    self.generator,
                    (
                        AutoModelForCausalLM, 
                        AutoPeftModelForCausalLM,
                        PeftModelForCausalLM,
                    )
                )
            logger.debug(f"using generator of type: {type(self.generator)}, decoder_only: {self.decoder_only}")
            return
        try:
            auto_cls = (
                AutoPeftModelForSeq2SeqLM 
                if self.is_peft_model 
                else AutoModelForSeq2SeqLM
            )
            self.generator = auto_cls.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                quantization_config=self.quantization_config,
            )
            self.decoder_only = False
        except ValueError:
            auto_cls = (
                AutoPeftModelForCausalLM
                if self.is_peft_model 
                else AutoModelForCausalLM
            )
            self.generator = auto_cls.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                quantization_config=self.quantization_config,
            )
            self.decoder_only = True
        self.generator = self.generator.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    async def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        # state = self.template % state
        state = self.template.format(state=state)
        logger.debug(state)
        if (
            self.decoder_only 
            and self.tokenizer.encode(state) > self.max_inp_seq_len
        ):
            # decoder only models won't generate tactics with truncated prompt
            # - it just tries to complete the prompt
            # - instead, just stop here
            return []
        tokenized_state = self.tokenizer(
            state, 
            max_length=self.max_inp_seq_len, 
            truncation=True, 
            return_tensors="pt",
        )
        state_ids = tokenized_state.input_ids.to(self.device)
        state_mask = tokenized_state.attention_mask.to(self.device)

        # Generate tactic candidates using beam search.
        num_beams = num_samples if self.use_beam_search else 1
        output = self.generator.generate(
            input_ids=state_ids,
            attention_mask=state_mask,
            # .generate expects one of max_length or max_new_tokens
            # max_length=self.max_oup_seq_len,
            max_new_tokens=self.max_new_tokens,
            num_beams=num_beams,
            length_penalty=self.length_penalty,
            eos_token_id=self.end_of_step_token_id,
            do_sample=False,
            num_return_sequences=num_samples,
            early_stopping=False,
            output_scores=True,
            output_logits=True,
            return_dict_in_generate=True,
        )

        # Return the output.
        raw_output_text = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )
        if self.use_beam_search:
            raw_scores = output.sequences_scores.tolist()
        else:
            # when not using beam_search, generate returns a Generate(Arch)Output
            # which unlike GenerateBeam(Arch)Output does not have sequences_scores
            # we have to compute the scores manually
            if self.decoder_only:
                transition_scores = self.generator.compute_transition_scores(
                    output.sequences, output.logits, normalize_logits=True
                )
                pad_mask = (
                    output.sequences[state_ids.shape[1]:]
                ).eq(self.tokenizer.pad_token_id)
                transition_scores[pad_mask] = 0
                raw_scores = transition_scores.sum(dim=1).tolist
            else:
                raise ValueError("non-beam search not yet supported for seq2seq models")

        output_text = []
        output_score = []

        for j in range(num_samples):
            t = remove_marks(raw_output_text[j])
            if self.decoder_only and t.startswith(state):
                # skip prompt
                t = t[len(state):]
                # end at next newline
                next_newline = t.find("\n")
                if next_newline != -1:
                    t = t[:next_newline]
            if t not in output_text:
                output_text.append(t)
                output_score.append(raw_scores[j])

        return list(zip_strict(output_text, output_score))


class RetrievalAugmentedGenerator(TacticGenerator):

    def __init__(
        self,
        gen_path: str,
        ret_path: str,
        indexed_corpus_path: str,
        device,
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        length_penalty: float,
        max_num_retrieved: int,
    ) -> None:
        self.gen_path = gen_path
        self.ret_path = ret_path
        self.indexed_corpus_path = indexed_corpus_path
        self.device = device
        self.max_inp_seq_len = max_inp_seq_len
        self.max_oup_seq_len = max_oup_seq_len
        self.length_penalty = length_penalty
        self.max_num_retrieved = max_num_retrieved
        self.hf_gen = HuggingFaceGenerator(
            gen_path, device, max_inp_seq_len, max_oup_seq_len, length_penalty
        )

    def initialize(self) -> None:
        self.hf_gen.initialize()
        self.retriever = PremiseRetriever.load_hf(
            self.ret_path, self.max_inp_seq_len, self.device
        )
        self.retriever.load_corpus(self.indexed_corpus_path)

    async def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        retrieved_premises, _ = self.retriever.retrieve(
            state,
            file_path,
            theorem_full_name,
            theorem_pos,
            self.max_num_retrieved,
        )
        aug_state = format_augmented_state(
            state, retrieved_premises, self.max_inp_seq_len
        )
        return await self.hf_gen.generate(
            aug_state, file_path, theorem_full_name, theorem_pos, num_samples
        )


class VllmGenerator(TacticGenerator):
    def __init__(self, vllm_actor, template: str = "[GOAL]\n%s\n[PROOFSTEP]\n") -> None:
        self.vllm_actor = vllm_actor
        self.template = template

    def initialize(self) -> None:
        pass

    async def generate(
        self,
        state: str,
        file_path: str,
        theorem_full_name: str,
        theorem_pos: Pos,
        num_samples: int,
    ) -> List[Tuple[str, float]]:
        prompt = self.template % state
        response = await self.vllm_actor.generate.remote(prompt, num_samples)
        return [
            (remove_marks(x.text).strip(), x.cumulative_logprob)
            for x in response.outputs
        ]
