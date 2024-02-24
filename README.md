# NTP Project

## Overview
**Broad Idea.** GFN objective provides a new way to sample compositional objects (e.g. LLM rationale/theorem proof) with key benefits of (1) better diversity (2) better generalizability. In the ICLR paper, the authors demonstrate that a simple reasoning (1 sentence for a binary classification) can be implemented with this objective. *We would like to extend this idea to more complex problems/inference techniques.*

**Implementation Challenge.** CoT reasoning results in a large number of tokens (long trajectory). Together with high inference cost of LLM, and the inherent difficulty of stabilizing RL training, this makes the training difficult.

We turn our attention to NTP where the LLM only needs to generate tactics (costing fewer tokens) and then the deterministic Lean kernel (or an arbitrary proof assistant) unfold the tactic.

The Lean kernel acts as a rigid reward function. It gives a binary signal if a completed proof is correct or not. However for GFN tuning, we would like to be able to evaluate partial sequences. One way is to train a verifier model.

Through the V-STaR project experiments, it was found that DPO produces a stronger verifier than through standard MLE training, but it also contains spurious modes. Since GFN-tuning is contingent on the reward being robust, we need some way to improve on this reward.

**Let's try making an ensemble of verifiers.** The idea is that multiple verifiers trained in ensemble are less likely to collapse the modes/exploit the same artifacts. Hopefully this produces a reward space rich enough to make the GFN-tuning process effective.

## To-do List

### Training Resources
Dataset
- [LeanDojo](https://leandojo.org) benchmark
Base Models
- [Llemma-7B](https://huggingface.co/EleutherAI/llemma_7b) (CodeLlama derivative)
- [ReProver](https://github.com/lean-dojo/ReProver) (LeanDojo RAG model)
Training Infrastructure
- [HuggingFace's DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)
	- expected dataset format
		- entries should be named 
			- `prompt`,`chosen`, and `rejected`
			- these are just sentences
	- expected model format
		- expects: `AutoModelForCausalLM`
			- this means that reprover is out of the picture
			- correction: https://github.com/huggingface/trl/pull/586
				- seq2seq support for DPO trainer has been added
				- still need to check which release this is in

### What is the format of LeanDojo's data
- `corpus.jsonl` contains the definition of premises
	- What are premises?
		- premises are theorems and definitions defined in some large math library
		- it's just extra information about math that might be known and used for proofs
		- things like `mod_self` (a theorem stating $n \% n = 0$) or `gcd_zero_left` (a theorem stating that $\text{gcd}(0,k) = k$)
	- Each line in the jsonl corresponds to a Lean file
		- it contains the path to the original lean file
		- a list of files that are imported
		- and then a list of premises
			- full_name
			- code
			- start/end location in file
			- also its "kind" (e.g. class, definition, etc.)
- Theorems: are split into training/validation/testing as json files
	- includes url, commit (these are mined from Github), theorem name, start/end in file and **traced tactics**: `list[traced_tactic]`
	- What's in a traced tactic?
		- tactic: raw tactic
		- state_before
		- state_after
		- annotated_tactic
	- What's an annotated tactic?
		- `annotated_tactic: tuple[annotated_statement, list[premise_provenance]]`
		- some tactics require some inputs/arguments/parameters (premises)
			- e.g. rewrite could require a theorem you want to substitute in
		- annotated statement wrap premises used in a tactic statement with HTML-style tags `<a></a>`
		- after the annotated  statement there is a provenance list which describes where each premise mentioned in the tactic comes from describing:
			- the fully specified name: `full_name`
			- the file it comes from: `def_path`
			- the start position of its definition
Note that there are proofs that don't use tactic-style completely.
Thus there are some examples where concatenating the tactics doesn't complete the original theorem's proof.
### Where to get Verifier Training Data?
- First of all, is there publicly available proof generation data online
	- Seems that neither Llemma nor ReProver have such data published on their repo
	- Could reach out to them or try to generate it ourselves
- Maybe we can consider a better model? or the one closer to 50-50?
	- Comparing performance:
		- LeanDojo mainly publishes results on its own new benchmark, but they also have baseline results for ReProver on MiniF2F and ProofNet
			- MiniF2F: Pass@1 = 26.5% on test split (with RAG)
			- ProofNet: Pass@1 = 13.8% first reported result on ProofNet apparently
		- Llemma only evaluates on MiniF2F and ProofNet, so we have to compare on these categories
		- ReProver has an appendix section explaining why comparison with other baselines is difficult/impossible

![[Assets/minif2f_comparison.png]]

Basically they have basically equivalent performance when ReProver has RAG and searches 64 next tactics per iteration and Llemma searches 32 next tactics per iteration.

Best-First-Search is the standard inference algorithm from next-tactic generators


## 23.02.2024 notes
There's this problem where the evaluation code for the models depends on Pytorch Lightning checkpoints but the models are only publicly available on HuggingFace.

I started writing code to manually instantiate the Lightning model classes using the HuggingFace models, but then realized that I could probably just make dummy classes with the same inference interface. 

New problem: I also need the retriever index to get the same performance

I'm leaning more towards using Llemma-7B for simplicity, so let's actually pivot and do that, despite the fact that our inference throughput will be much much worse.

I'm trying to mock the following

```
python retrieval/index.py --ckpt_path PATH_TO_RETRIEVER_CHECKPOINT --corpus-path data/leandojo_benchmark/corpus.jsonl --output-path PATH_TO_INDEXED_CORPUS
python retrieval/index.py --ckpt_path PATH_TO_RETRIEVER_CHECKPOINT --corpus-path data/leandojo_benchmark_4/corpus.jsonl --output-path PATH_TO_INDEXED_CORPUS
# Do it separately for two data splits.

python prover/evaluate.py --data-path data/leandojo_benchmark/random/  --ckpt_path PATH_TO_REPROVER_CHECKPOINT --indexed-corpus-path PATH_TO_INDEXED_CORPUS --split test --num-cpus 8 --with-gpus
python prover/evaluate.py --data-path data/leandojo_benchmark_4/random/  --ckpt_path PATH_TO_REPROVER_CHECKPOINT --indexed-corpus-path PATH_TO_INDEXED_CORPUS --split test --num-cpus 8 --with-gpus
# Do it separately two data splits.
```

Status: IN PROGRESS
#### Other papers to look into
I asked one of the authors of LeanDojo for the outputs to their experiments. They told me to contact another author but they also gave a recommendation: 
```
Oh also, just a side note, despite LeanDojo being pretty recent, I do recall seeing even newer published / unpublished works that claim to beat LeanDojo on accuracy numbers (not to a great deal though). So if you later conduct experiments for your project maybe remember to compare to newer SoTAs.
```

Per their recommendation, I checked Semantic Scholar to find papers that have cited LeanDojo.
Next is to check if there is a 
- LEGO-Prover: https://arxiv.org/pdf/2310.00656.pdf
- Enhanced NTP via Data Augmentation and Dynamic Sampling: https://arxiv.org/pdf/2312.14188.pdf
- ICL Agent for Formal NTP https://arxiv.org/pdf/2310.04353.pdf
- Lyra: Dual Correction for NTP https://arxiv.org/pdf/2309.15806.pdf
- Structure-Aware Repr for Dependent Types https://arxiv.org/pdf/2402.02104.pdf
- LLMSTEP: https://arxiv.org/pdf/2310.18457.pdf
- LM MathAgent (withdrawn?) https://arxiv.org/abs/2312.08926
- Tactician's Web: network/graph of formal math knowledge: https://arxiv.org/pdf/2401.02950.pdf
- Understanding Reasoning via Path Aggregation: https://arxiv.org/pdf/2402.03268.pdf
- InternLM-Math: verifiable reasoning: https://arxiv.org/pdf/2402.06332.pdf