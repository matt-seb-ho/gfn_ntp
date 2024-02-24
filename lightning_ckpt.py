# creating pytorch lightning checkpoints from HuggingFace models
import torch
from typing import Union, List
from transformers import (
    AutoTokenizer, AutoModelForTextEncoding, AutoModelForSeq2SeqLM
)

# --------------------------------------------------------------------------------
# download models from HuggingFace

hf_ret_path = "kaiyuy/leandojo-lean4-retriever-byt5-small" # lean3 <-> lean4
retriever_tk = AutoTokenizer.from_pretrained(hf_ret_path)
retriever = AutoModelForTextEncoding.from_pretrained(hf_ret_path)

hf_rag_path = "kaiyuy/leandojo-lean4-retriever-tacgen-byt5-small"
rag_tk = AutoTokenizer.from_pretrained(hf_rag_path)
rag = AutoModelForSeq2SeqLM.from_pretrained(hf_rag_path)


# --------------------------------------------------------------------------------
# models to instantiate
# class RetrievalAugmentedGenerator(TacticGenerator, pl.LightningModule):
#     def __init__(
#         self,
#         model_name: str,
#         lr: float,
#         warmup_steps: int,
#         num_beams: int,
#         eval_num_retrieved: int,
#         eval_num_cpus: int,
#         eval_num_theorems: int,
#         max_inp_seq_len: int,
#         max_oup_seq_len: int,
#         length_penalty: float = 0.0,
#         ret_ckpt_path: Optional[str] = None,
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters()
#         self.lr = lr
#         self.warmup_steps = warmup_steps
#         self.num_beams = num_beams
#         self.length_penalty = length_penalty
#         self.eval_num_retrieved = eval_num_retrieved
#         self.eval_num_cpus = eval_num_cpus
#         self.eval_num_theorems = eval_num_theorems
#         self.max_inp_seq_len = max_inp_seq_len
#         self.max_oup_seq_len = max_oup_seq_len
# 
#         if ret_ckpt_path is None:
#             logger.info("Without retrieval")
#             self.retriever = None
#         else:
#             logger.info(f"Loading the retriever from {ret_ckpt_path}")
#             self.retriever = PremiseRetriever.load(
#                 ret_ckpt_path, self.device, freeze=True
#             )
# 
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.generator = T5ForConditionalGeneration.from_pretrained(model_name)
# 
#         self.topk_accuracies = dict()
#         for k in range(1, num_beams + 1):
#             acc = TopkAccuracy(k)
#             self.topk_accuracies[k] = acc
#             self.add_module(f"top{k}_acc_val", acc)
# 
# class PremiseRetriever(pl.LightningModule):
#     def __init__(
#         self,
#         model_name: str,
#         lr: float,
#         warmup_steps: int,
#         max_seq_len: int,
#         num_retrieved: int = 100,
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters()
#         self.lr = lr
#         self.warmup_steps = warmup_steps
#         self.num_retrieved = num_retrieved
#         self.max_seq_len = max_seq_len
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.encoder = T5EncoderModel.from_pretrained(model_name)
#         self.embeddings_staled = True


# --------------------------------------------------------------------------------
# example retriever usage
# from https://github.com/lean-dojo/ReProver/tree/main?tab=readme-ov-file#premise-retriever
# 
# state = "n : ℕ\n⊢ gcd n n = n"
# premises = [
#   "<a>vsub_eq_zero_iff_eq</a> @[simp] lemma vsub_eq_zero_iff_eq {p1 p2 : P} : p1 -ᵥ p2 = (0 : G) ↔ p1 = p2",
#   "<a>is_scalar_tower.coe_to_alg_hom'</a> @[simp] lemma coe_to_alg_hom' : (to_alg_hom R S A : S → A) = algebra_map S A",
#   "<a>polynomial.X_sub_C_ne_zero</a> theorem X_sub_C_ne_zero (r : R) : X - C r ≠ 0",
#   "<a>forall_true_iff</a> theorem forall_true_iff : (α → true) ↔ true",
#   "def <a>nat.gcd</a> : nat → nat → nat\n| 0        y := y\n| (succ x) y := have y % succ x < succ x, from mod_lt _ $ succ_pos _,\n                gcd (y % succ x) (succ x)",
#   "@[simp] theorem <a>nat.gcd_zero_left</a> (x : nat) : gcd 0 x = x",
#   "@[simp] theorem <a>nat.gcd_succ</a> (x y : nat) : gcd (succ x) y = gcd (y % succ x) (succ x)",
#   "@[simp] theorem <a>nat.mod_self</a> (n : nat) : n % n = 0",
# ]  # A corpus of premises to retrieve from.
# 
# @torch.no_grad()
# def encode(s: Union[str, List[str]]) -> torch.Tensor:
#     """Encode texts into feature vectors."""
#     if isinstance(s, str):
#         s = [s]
#         should_squeeze = True
#     else:
#         should_squeeze = False
#     tokenized_s = tokenizer(s, return_tensors="pt", padding=True)
#     hidden_state = model(tokenized_s.input_ids).last_hidden_state
#     lens = tokenized_s.attention_mask.sum(dim=1)
#     features = (hidden_state * tokenized_s.attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
#     if should_squeeze:
#       features = features.squeeze()
#     return features
# 
# @torch.no_grad()
# def retrieve(state: str, premises: List[str], k: int) -> List[str]:
#     """Retrieve the top-k premises given a state."""
#     state_emb = encode(state)
#     premise_embs = encode(premises)
#     scores = (state_emb @ premise_embs.T)
#     topk = scores.topk(k).indices.tolist()
#     return [premises[i] for i in topk]
# 
# for p in retrieve(state, premises, k=4):
#     print(p, end="\n\n")



# --------------------------------------------------------------------------------
# example RAG usage
# from https://github.com/lean-dojo/ReProver/tree/main?tab=readme-ov-file#retrieval-augmented-tactic-generator
#
# state = "n : ℕ\n⊢ gcd n n = n"
# retrieved_premises = [
#   "def <a>nat.gcd</a> : nat → nat → nat\n| 0        y := y\n| (succ x) y := have y % succ x < succ x, from mod_lt _ $ succ_pos _,\n                gcd (y % succ x) (succ x)",
#   "@[simp] theorem <a>nat.mod_self</a> (n : nat) : n % n = 0",
# ]
# input = "\n\n".join(retrieved_premises + [state])
# print("------ INPUT ------\n", input)
# tokenized_input = tokenizer(input, return_tensors="pt", max_length=2300, truncation=True)
# 
# # Generate a single tactic.
# tactic_ids = model.generate(tokenized_input.input_ids, max_length=1024)
# tactic = tokenizer.decode(tactic_ids[0], skip_special_tokens=True)
# print("\n------ OUTPUT ------")
# print(tactic, end="\n\n")
# 
# # Generate multiple tactics via beam search.
# tactic_candidates_ids = model.generate(
#     tokenized_input.input_ids,
#     max_length=1024,
#     num_beams=4,
#     length_penalty=0.0,
#     do_sample=False,
#     num_return_sequences=4,
#     early_stopping=False,
# )
# tactic_candidates = tokenizer.batch_decode(
#     tactic_candidates_ids, skip_special_tokens=True
# )
# for tac in tactic_candidates:
#     print(tac)