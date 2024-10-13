# Proof Flow: GFlowNet Tuning for Neural Theorem Proving (NTP)

We conduct a preliminary study examining the potential of applying [GFlowNet Tuning](https://github.com/GFNOrg/gfn-lm-tuning) to formal reasoning, specifically, neural theorem proving (NTP). 

Our study is motivated by (1) the observation that standard reasoning benchmarks (e.g. GSM8K) are increasingly overfitted against and do not perfectly evaluate model performance in real-world frontier use cases, and (2) the development of GFlowNet Tuning as a well-principled approach to improving sampling diversity and search performance by "amortizing" the cost of sampling more completions at inference time into a post-training phase.

Preliminary results and discussion using base model [ReProver](https://github.com/lean-dojo/ReProver) and a subset of the [LeanDojo benchmark dataset](https://github.com/lean-dojo/LeanDojo) can be found in the paper (`workshop_paper.pdf`).

## Instructions
Minimal installation:
1. clone this repository
2. install [Lean Dojo dependencies](https://leandojo.readthedocs.io/en/latest/getting-started.html#requirements)
3. install packages: `pip install -r requirements.txt`
4. update paths in `gfn_ntp/configs/paths/default.yaml` to point to correct directories

5. prepare dataset: select one of the following options 
	- download `shuffled_balanced1k.json`, `val20.json` to `gfn_ntp/data/` from [here](https://drive.google.com/drive/folders/1q_g59GBik3z8SngREnCC5iclnfVEB3gk?usp=drive_link)
	- download raw files from LeanDojo benchmark and use the filtering script (`python -m proof_flow.scripts.data_preprocessing.filter_theorems`) (*DETAILED INSTRUCTIONS COMING SOON*)

6. start training with `python -m proof_flow.scripts.gfn_tuning.train`


---
#### TODO: 
- add OpenReview link (currently down)
	- alternatively, upload arxiv version
- add detailed instructions for filtering data
