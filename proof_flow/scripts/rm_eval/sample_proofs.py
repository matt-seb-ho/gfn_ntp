from typing import Optional
import argparse
import json
from pathlib import Path
from vllm import LLM, SamplingParams
from tqdm import tqdm

from proof_flow.src.utils import get_config, repo_root
from lean_dojo import LeanGitRepo, Pos, Theorem
from .proof_search import ProverActor, BestFirstSearchProver
from .search_tree import Status
from .tactic_generator import TacticGenerator, HuggingFaceGenerator

"""
given: 
- set of theorems and tactic proofs
goal:
- RM evaluation set: triples of form (proof state, correct tactic, incorrect tactic)
approach:
- use a current SOTA model to sample proofs
- SOTA solve rate on miniF2F is still ~50% (32-128) attempts, so we'll still get incorrect proofs
- pass all attempts through Lean verifier
- for incorrect attempts, we get a correct tactic t_i if t_1,...t_i does not appear in a correct proof

first step:
- write code to sample proofs from SOTA model (deep seek prover 1.5)
- use vLLM
"""

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


def sample_proofs(
    thm_dicts: list | dict,
    llm: LLM,
    sampling_params: SamplingParams,
    use_tqdm: bool = False,
) -> list[list[str]]:
    """
    given:  theorem info
    return: list of {sample_num} proofs
    """
    if isinstance(thm_dicts, list):
        model_inputs = [cot_prompt(thm_dict) for thm_dict in thm_dicts]
    else:
        model_inputs = [cot_prompt(thm_dict) for thm_dict in thm_dicts.values()]
    
    # In the DeepSeek code base they use SamplingParams(n=1) inside a for loop 
    # because the requests get routed to multiple generator/gpu processes.
    # In our case, we can just use the n parameter as intended
    model_outputs = llm.generate(
        model_inputs,
        sampling_params,
        use_tqdm=use_tqdm,
    )
    # outputs: list[RequestOutput], RequestOutput.outputs: list[CompletionOutput]
    # reference line: `outputs = [self.output_func(_output.outputs[0].text) for _output in model_outputs]`
    outputs = []
    for prompt_outputs in model_outputs:
        outputs.append([
            post_process_output(output.text) 
            for output in prompt_outputs.outputs
        ])
    return outputs
     
        
def sample_proofs_with_best_first_search(
    thm_dicts: list | dict,
    tactic_generator: TacticGenerator,
    timeout: int,
    max_expansions: Optional[int] = None,
    num_sampled_tactics: int = 1,
    use_tqdm: bool = False,
) -> list[list[str]]:
    """
    Sample proofs using BestFirstSearchProver instead of an LLM.
    """
    # Initialize the prover with the given tactic generator
    prover = BestFirstSearchProver(
        tac_gen=tactic_generator,
        timeout=timeout,
        max_expansions=max_expansions,
        num_sampled_tactics=num_sampled_tactics,
        debug=True,
    )

    proofs = []
    if isinstance(thm_dicts, list):
        thms = thm_dicts
    else:
        thms = thm_dicts.values()

    # Use tqdm for progress bar if use_tqdm is True
    iterator = tqdm(thms, desc="Sampling Proofs w/ Best First Search") if use_tqdm else thms

    for thm_data in iterator:
        # Set up theorem and repo information
        repo = LeanGitRepo(thm_data['url'], thm_data['commit'])
        theorem = Theorem(
            repo=repo,
            file_path=thm_data['file_path'],
            full_name=thm_data['full_name'],
        )
        pos = Pos(thm_data['start'][0], thm_data['start'][1])

        # Use the prover to search for a proof
        result = prover.search(repo, theorem, pos)

        # Collect the proof if found
        if result and result.status == Status.PROVED:
            proofs.append(result.proof)
        else:
            proofs.append([])  # No proof found or other status

    return proofs
     
        
def cot_prompt(data):
    return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
        header=data.get('header', LEAN4_DEFAULT_HEADER),
        informal_prefix=data.get('informal_prefix', str()),
        formal_statement=data['formal_statement'],
    )


def post_process_output(output):
    _find_idx = output.find("```")
    return output[:_find_idx] if _find_idx >= 0 else output
    
    
if __name__ == "__main__":
    # parser for taking non-default config file
    psr = argparse.ArgumentParser()
    psr.add_argument("--config", type=str, default="rm_eval")
    args = psr.parse_args()

    # values from DeepSeek-Prover-V1.5/configs/sampling.py
    # initialization from DeepSeek-Prover-V1.5/prover/workers/generator.py
    # - except seed which was `seed = int(time.time()) % 1000 + (self.node_rank * 8 + self.local_rank) * 1000`
    cfg = get_config(config_name=args.config)

    theorem_file = repo_root() / cfg.input_file
    with open(theorem_file) as f:
        input_data = json.load(f)
    # subset data for sanity checking/testing phase
    if cfg.input_limit is not None:
        keys = list(input_data.keys())
        input_data = {keys[i]: input_data[keys[i]] for i in range(cfg.input_limit)}


    if cfg.use_BFS:
        # define a tactics generator with the given model
        tactic_generator = HuggingFaceGenerator(
            model_path=cfg.data_generation.llm.model, 
            device=cfg.data_generation.device,
            max_inp_seq_len=cfg.data_generation.max_inp_seq_len,
            max_oup_seq_len=cfg.data_generation.max_oup_seq_len,
            length_penalty=cfg.data_generation.length_penalty,
            template=cfg.data_generation.template,
        )
        
        # Test the max expansion (10, maybe None) and timeout (10) parameters for the BestFirstSearchProver see how many proofs are found
        # Sample 30 and evaluated if proof correctness. You can check how good it is tactic at a time (accuracy, try to get accuracy > 50%) 
        # max_inp_seq_len = max length of the theorems (already removed top 5%)
        # max_oup_seq_len = max number of new tokens (30). 
        
        # get proofs with Best First Search and the defined tactic generator
        proofs = sample_proofs_with_best_first_search(
            input_data, 
            tactic_generator, 
            timeout=cfg.data_generation.timeout, # in seconds, the amount of time it takes for the dojo to find a proof
            max_expansions=cfg.data_generation.max_expansions, 
            num_sampled_tactics=cfg.data_generation.sampling_params.n, 
            use_tqdm=cfg.use_tqdm
        )
    else:
        sampling_params = SamplingParams(**cfg.data_generation.sampling_params)
        llm = LLM(**cfg.data_generation.llm)
        # get proofs
        proofs = sample_proofs(input_data, llm, sampling_params, use_tqdm=cfg.use_tqdm)
    

    # write to file
    output_file = repo_root() / cfg.output_file
    with open(output_file, "w") as f:
        json.dump(proofs, f, indent=4)
        if cfg.use_BFS:
            print(f"sampled proofs with Best First Search written to {output_file}")
        else:
            print(f"sampled proofs written to {output_file}")


    
# --------------------------------------------------------------------
# code reference
# source: https://github.com/deepseek-ai/DeepSeek-Prover-V1.5


# SAMPLING CONFIG ------------------------------
# from prover.utils import AttrDict
# from prover.algorithms import Sampling


# # dataset
# data_path = 'datasets/minif2f.jsonl'
# data_split = ['valid', 'test']
# data_repeat = 1

# # verifier
# lean_max_concurrent_requests = 64
# lean_memory_limit = 10
# lean_timeout = 300

# # model
# batch_size = 32
# model_path = 'deepseek-ai/DeepSeek-Prover-V1.5-RL'
# model_args = AttrDict(
#     mode='cot',  # `cot` or `non-cot`
#     temperature=1,
#     max_tokens=2048,
#     top_p=0.95,
# )

# # algorithm
# n_search_procs = 64
# sampler = dict(
#     algorithm=Sampling,
#     sample_num=128,
#     log_interval=32,
# )

# SAMPLER.SAMPLE ------------------------------
    # def sample(self, data, **kwargs):
    #     request_id_list = [
    #         self.scheduler.generator_submit_request(
    #             # add few-shot prompts
    #             self._preprocess_data(data),
    #         ) for _ in range(self.sample_num)
    #     ]
    #     for _idx, request_id in enumerate(request_id_list):
    #         outputs = self.scheduler.generator_get_request_outputs(request_id)
    #         yield outputs, self._post_sample_info(cost=_idx+1)
    #         if _idx + 1 < self.sample_num and (_idx + 1) % self.log_interval == 0:
    #             self.process_print('Progress: {} / {}'.format(
    #                 _idx + 1, self.sample_num
    #             ))

# GENERATOR PROCESS RUN ------------------------------
        # llm = LLM(model=self.model_path, max_num_batched_tokens=8192, seed=seed, trust_remote_code=True)
# ...
        # self.sampling_params = SamplingParams(
        #     temperature=args.temperature,
        #     max_tokens=args.max_tokens,
        #     top_p=args.top_p,
        #     n=1,
        # )
# ...
            # model_inputs = [
            #     ''.join([
            #         item.get('_extra_header', str()),
            #         self.prompt_func(item),
            #         item.get('_extra_prompt', str()),
            #     ]) for _, _, item in inputs
            # ]
            # model_outputs = llm.generate(
            #     model_inputs,
            #     self.sampling_params,
            #     use_tqdm=False,
            # )
            # outputs = [self.output_func(_output.outputs[0].text) for _output in model_outputs]

# PROMPTING METHODS ------------------------------
# LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

# def non_cot_prompt(data):
#     return "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
#         header=data.get('header', LEAN4_DEFAULT_HEADER),
#         informal_prefix=data.get('informal_prefix', str()),
#         formal_statement=data['formal_statement'],
#     )

# def non_cot_few_shot_prompt(data):
#     return "Complete the following Lean 4 code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}{formal_proof}\n```\n\n\n".format(
#         header=data.get('header', LEAN4_DEFAULT_HEADER),
#         informal_prefix=data.get('informal_prefix', str()),
#         formal_statement=data['formal_statement'],
#         formal_proof=data['formal_proof'],
#     )

# def cot_prompt(data):
#     return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
#         header=data.get('header', LEAN4_DEFAULT_HEADER),
#         informal_prefix=data.get('informal_prefix', str()),
#         formal_statement=data['formal_statement'],
#     )

# def cot_few_shot_prompt(data):
#     return "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}{formal_proof}\n```\n\n\n".format(
#         header=data.get('header', LEAN4_DEFAULT_HEADER),
#         informal_prefix=data.get('informal_prefix', str()),
#         formal_statement=data['formal_statement'],
#         formal_proof=data['formal_proof'],
#     )

# def post_process_output(output):
#     _find_idx = output.find("```")
#     return output[:_find_idx] if _find_idx >= 0 else output

# MODEL_FORMAT = dict(
#     non_cot=dict(prompt=non_cot_prompt, output=post_process_output, few_shot=non_cot_few_shot_prompt),
#     cot=dict(prompt=cot_prompt, output=post_process_output, few_shot=cot_few_shot_prompt),
# )
