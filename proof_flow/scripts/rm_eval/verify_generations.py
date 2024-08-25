import json
import re
from tqdm import tqdm
from proof_flow.src.utils import get_config, repo_root, prepare_environment_for_lean_dojo


prepare_environment_for_lean_dojo()
from lean_dojo import ( # isort: skip
    Dojo, LeanGitRepo, Theorem, ProofFinished, TacticState, LeanError, ProofGivenUp
)

import re

"""
given: 
- set of theorems and tactic proofs
- a set of generated proof completions for each theorem
goal:
- RM evaluation set: triples of form (proof state, correct tactic, incorrect tactic)
approach:
- use a current SOTA model to sample proofs
- SOTA solve rate on miniF2F is still ~50% (32-128) attempts, so we'll still get incorrect proofs
- pass all attempts through Lean verifier
- for incorrect attempts, we get a correct tactic t_i if t_1,...t_i does not appear in a correct proof

second step:
- use LeanDojo to verify the generated proofs
- LeanDojo provides interaction via `run_tac`
- we need to extract the individual tactics from the generated proofs (which contain comments)
"""

def extract_tactics_from_completion(completion: str) -> list[str]:
    """
    given: proof completion
    return: list of tactics
    approach:
    - first use regex to remove multi-line comments
    - separate back out into lines
    - remove single line comments and empty lines
    - assume each line is a tactic
    """
    # Step 1: Remove multi-line comments
    completion = re.sub(r'/-.*?-/', '', completion, flags=re.DOTALL)

    # Step 2: Split into lines
    lines = completion.splitlines()

    # Step 3: Remove single-line comments and empty lines
    tactics = []
    for line in lines:
        # Remove single-line comments
        line = re.sub(r'--.*', '', line).strip()
        # If the line is not empty after stripping, consider it a tactic
        if line:
            tactics.append(line)
    
    return tactics


def verify_proof_candidates(
    thm_info: dict,
    candidates: list[str],
    repo: LeanGitRepo,
) -> list[tuple[bool, list[dict]]]:
    """
    given: thm info and repo to initialize the dojo, list of proof candidates
    goal: for each candidate, whether the proof is correct (bool) and traced tactics (tactic, state_before, state_after)
    approach:
    - for each candidate
        - get tactic list
        - run tactic list through LeanDojo, recording the intermediate tactic states
        - if final state is "no goals" then the proof is correct
    """
    theorem = Theorem(repo, thm_info["file_path"], thm_info["full_name"])
    candidate_tactics = [extract_tactics_from_completion(candidate) for candidate in candidates]
    results = []

    with Dojo(theorem) as (dojo, init_state):
        for sequence in candidate_tactics:
            state = init_state
            trace = []
            early_exit = None
            for tactic in sequence:
                trace_entry = {
                    "state_before": state.pp,
                    "tactic": tactic,
                    "state_after": None,
                    "message": None,
                }
                result = dojo.run_tac(state, tactic)
                if isinstance(result, LeanError):
                    trace_entry["message"] = result.error
                    early_exit = "error"
                elif isinstance(result, ProofFinished):
                    trace_entry["state_after"] = "no goals"
                    trace_entry["message"] = result.message
                    early_exit = "no goals"
                elif isinstance(result, TimeoutError):
                    trace_entry["message"] = "timeout"
                    early_exit = "timeout"
                elif isinstance(result, ProofGivenUp):
                    trace_entry["message"] = "given up"
                    early_exit = "given up"
                else:
                    # result is a TacticState
                    state = result
                    trace_entry["state_after"] = state.pp
                trace.append(trace_entry)
                if early_exit:
                    break
            valid_proof = (trace[-1]["state_after"] == "no goals")
            results.append((valid_proof, trace))
    return results


def verify_batch(
    thm_dicts: dict,
    candidates: list[list[str]],
) -> list[list[tuple[bool, list[dict]]]]:
    """
    given: list of theorem info, list of proof candidates, and LeanGitRepo
    goal: for each theorem, for each candidate, whether the proof is correct (bool) and traced tactics (tactic, state_before, state_after)
    approach:
    - for each theorem
        - verify the proof candidates
    """
    thm0 = thm_dicts[0] if isinstance(thm_dicts, list) else next(iter(thm_dicts.values()))
    repo = LeanGitRepo(thm0["url"], thm0["commit"])
    
    results = {}
    for (thm_idx, thm_info), candidate in tqdm(zip(thm_dicts.items(), candidates), total=len(thm_dicts)):
        results[thm_idx] = verify_proof_candidates(thm_info, candidate, repo)
    return results

    

if __name__ == "__main__":
    cfg = get_config(config_name="rm_eval")

    # load the generated proofs
    proof_file = repo_root() / cfg.output_file
    with open(proof_file) as f:
        proofs = json.load(f)

    # load the theorem info
    thm_file = repo_root() / cfg.input_file
    with open(thm_file) as f:
        thm_dicts = json.load(f)
    
    # test tactic extraction
    # completion00 = proofs[0][0]
    # tactics = extract_tactics_from_completion(completion00)
    # print(completion00)
    # print("\n\n --- \n")
    # print(tactics)
    # test result: good enough

    results = verify_batch(thm_dicts, proofs)

    # gather stats
    """
    what steps do I care about?
    per thm:
    - if any of the attempts were correct
    - proportion of attempts per theorem that were correct
    total:
    - total proportion of attempts that were correct
    """
    thm_stats = {}
    total_correct_attempts = 0
    total_solved = 0
    for thm_idx, thm_res in zip(thm_dicts.keys(), results):
        thm_stats[thm_idx] = {
            "proved": False,
            "correct_attempts": 0
        }
        for correct, _ in thm_res:
            thm_stats[thm_idx]["proved"] = thm_stats[thm_idx]["proved"] or correct
            if correct:
                thm_stats[thm_idx]["correct_attempts"] += 1
                total_correct_attempts += 1
        if thm_stats[thm_idx]["proved"]:
            total_solved += 1
    
    print(f"Total solved: {total_solved}/{len(thm_dicts)}")
    print(f"Total correct attempts: {total_correct_attempts}/{len(proofs) * len(proofs[0])}")

    # write results to file
    with open(repo_root() / cfg.verification_results_file, 'w') as f:
        json.dump(results, f, indent=4)
