import sys
sys.path.append('/home/vincentzhu/gfn_ntp/proof_flow/scripts/')
import json
import pickle
import os
import math
import heapq
from collections import defaultdict, deque
from tqdm import tqdm

from proof_flow.src.constants import PROOF_COMPLETE_MESSAGE
from proof_flow.src.utils import get_config, repo_root

from proof_flow.scripts.prover.proof_search_tree import Status, DistributedProver

"""
given: 
- set of theorems and tactic proofs
- a set of generated proof completions for each theorem (STEP 1)
- verification results of proof completions (STEP 2)
    - for each theorem, for each completion...
      - whether the completion yields a correct proof
      - list[tactic_dict], where tactic_dict has form {state_before: str, tactic: str, state_after: str}
goal:
- RM evaluation set: triples of form (proof state, correct tactic, incorrect tactic)

current step (STEP 3): determine which sampled tactics are good/bad/neutral
- figure out every state's minimum distance to a complete proof
- tactics transition state_before to state_after
- measure tactic "goodness" by min_dist(state_before) - min_dist(state_after)
- if this is negative, the tactic is bad
- DESIGN DECISION: do we consider 0 change as bad?
"""

TESTING_MIN_DIST = False
TESTING_EXTRACT_RM_DATA = False

def get_tactic_label(dist_reduction):
    if dist_reduction > 0:
        return "positive"
    elif dist_reduction < 0:
        return "negative"
    return "neutral"


def extract_rm_data_from_thm(
    thm_dict: dict,
    verification_results: list[tuple[bool, list[dict]]],
) -> tuple[dict[str, dict], dict[str, dict]]:
    traces = [trace for _, trace in verification_results]
    traces.append(thm_dict["traced_tactics"])
    min_dist = get_minimum_distances(traces)
    state_to_tactic = defaultdict(lambda: {"positive": [], "negative": [], "neutral": []})
    for trace in traces:
        for entry in trace:
            dist_reduction = min_dist[entry["state_before"]] - min_dist[entry["state_after"]]
            entry["dist_reduction"] = str(dist_reduction) if dist_reduction == -1 * math.inf else dist_reduction
            label = get_tactic_label(dist_reduction)
            state_to_tactic[entry["state_before"]][label].append(entry)

    paired_data = {}
    for state, data in state_to_tactic.items():
        if len(data["positive"]) > 0 and len(data["negative"]) > 0:
            paired_data[state] = data

    return state_to_tactic, paired_data


def get_minimum_distances(
    traces: list[list[dict]],
    goal_state: str = PROOF_COMPLETE_MESSAGE,
) -> dict[str, int]:
    # construct state -> minimum distance from goal mapping
    # using a priority queue let's us ensure smallest distances are processed first 
    min_dist = defaultdict(lambda: math.inf)
    min_dist[goal_state] = 0
    pq = []
    for i, trace in enumerate(traces):
        pq.append((
            min_dist[trace[-1]["state_after"]],  # min heap sorting key
            i,                                   # trace idx
            len(trace) - 1,                      # entry idx
        ))
    heapq.heapify(pq)
  
    while pq:
        next_state_min_dist, trace_idx, entry_idx = heapq.heappop(pq)
        trace_entry = traces[trace_idx][entry_idx]
        state_before = trace_entry["state_before"]
        current_state_min_dist = min(min_dist[state_before], next_state_min_dist + 1)
        min_dist[state_before] = current_state_min_dist
        if entry_idx > 0:
            pq.append((current_state_min_dist, trace_idx, entry_idx - 1))

    return min_dist


def convert_state_seq_into_trace_fmt(lst):
    trace = []
    for i in range(len(lst) - 1):
        trace.append({
            "state_before": lst[i],
            "tactic": f"tactic{i}",
            "state_after": lst[i + 1],
        })
    return trace


def mine_stats(thm_dict, thm_results, verbose=False):
    state_to_tactic = extract_rm_data_from_thm(thm_dict, thm_results)

    pos_tac_count = 0
    neg_tac_count = 0
    neutral_tac_count = 0
    has_both_count = 0
    has_at_least_two = 0

    pn_training_pairs = 0
  
    for state, tactic_dict in state_to_tactic.items():
        pos_tac_count += len(tactic_dict["positive"])
        neg_tac_count += len(tactic_dict["negative"])
        neutral_tac_count += len(tactic_dict["neutral"])
        if len(tactic_dict["positive"]) > 0 and len(tactic_dict["negative"]) > 0:
            has_both_count += 1
            pn_training_pairs += len(tactic_dict["positive"]) * len(tactic_dict["negative"])

        # has at least two of 'positive', 'negative', 'neutral'
        non_empty_buckets = 0
        for bucket in ["positive", "negative", "neutral"]:
            if len(tactic_dict[bucket]) > 0:
                non_empty_buckets += 1
        if non_empty_buckets >= 2:
            has_at_least_two += 1
  
    stats = {
        "total_states": len(state_to_tactic),
        "mean_pos_tactics": pos_tac_count / len(state_to_tactic),
        "mean_neg_tactics": neg_tac_count / len(state_to_tactic),
        "neutral_tactics": neutral_tac_count,
        "states_with_both_pos_neg": has_both_count,
        "states_with_at_least_two": has_at_least_two,
        "pn_training_pairs": pn_training_pairs,
    }
    # print results
    if verbose:
        print(json.dumps(stats, indent=2))
    
  
def parse_proof_tree(proof_tree_root):
    """
    Given:
    - proof tree root node: InternalNode
    Task: 
    - Parse the proof tree with Breadth First Search to extract the tactics and their status
    Args:
    - proof_tree: InternalNode
    Returns:
    - list of negative tactics
    """
    tactics = []
    q = deque()
    neighbors = proof_tree_root.out_edges
    parsed_tactics = []
    # while not q.empty():
        
    # for node in neighbors:
        
    # return parsed_tactics
  
if __name__ == "__main__":
    """
    Given:
    - Sampled proof trees directory (test_proof_tactic_trees)
    - Each pickle file is associated with a theorem and its genrated proof tree 
    Task:
    - Iterates over the sampled_proofs directory using os.listdir()
    - For non theorem pickle files, skips them (log file and results file)
    - For each theorem pickle file, loads the proof tree for that theorem (11488 theorem pickle files)
    - PROCESS: parse_proof_tree, 
    - Saves the reward model data into a JSON file
    """
    
    cfg = get_config(config_name="rm_eval")
    
    proof_sampling_results_path = "/home/vincentzhu/gfn_ntp/data/test_proof_tactic_trees/metadata/0_test_proofs_v3_wfs_results.pickle"
    with open(proof_sampling_results_path, "rb") as f:
        proof_sampling_results = pickle.load(f)
    
    # Define the directory path
    # directory_path = cfg.proof_tactic_trees_dir
    directory_path = "/home/vincentzhu/gfn_ntp/data/test_proof_tactic_trees"

    # Iterate over each entry in the directory
    for entry in os.listdir(directory_path):
        # Get the full path of the entry
        full_path = os.path.join(directory_path, entry)
        
        if os.path.isdir(full_path):
            print(f"Directory: {full_path}")
            
        if os.path.isfile(full_path):
            with open(full_path, "rb") as f:
                proof_tree = pickle.load(f)
            print(type(proof_tree))
    
    
    # if TESTING_MIN_DIST:
    #     # testing phase
    #     test_trajectories1 = [
    #         [3, 2, 1, 0], 
    #         [1, 2, 3, 2, 0],
    #         [8, 1, 5, 6, 7, 0],
    #     ]
    #     test_trajectories1 = [convert_state_seq_into_trace_fmt(lst) for lst in test_trajectories1]

    #     expected_md = {0: 0, 1: 1, 2: 1, 3: 2, 5: 3, 6: 2, 7: 1, 8: 2}
    #     min_dist1 = get_minimum_distances(test_trajectories1, goal_state=0)

    #     # should be shuffle invariant
    #     tt2 = list(reversed(test_trajectories1))
    #     min_dist2 = get_minimum_distances(tt2, goal_state=0)
        
    #     assert min_dist1 == min_dist2, "oops, not trace order invariant"
    #     assert min_dist1 == expected_md, "oops, incorrect min distances"
  
    # cfg = get_config(config_name="rm_eval")
    # with open(repo_root() / cfg.verification_results_file) as f:
    #     verification_results = json.load(f) # dict[str, list[tuple[bool, list[dict]]]]
  
    # with open(repo_root() / cfg.input_file) as f:
    #     thm_dicts = json.load(f)
  
    # if TESTING_EXTRACT_RM_DATA:
    #     thm0_idx = next(iter(thm_dicts.keys()))
    #     thm0 = thm_dicts[thm0_idx]
    #     thm0_results = verification_results[thm0_idx]
    #     mine_stats(thm0, thm0_results, verbose=True)

    # pair_data = []
    # for thm_idx, thm_results in tqdm(verification_results.items()):
    #     thm_dict = thm_dicts[thm_idx]
    #     s2t, paired_only = extract_rm_data_from_thm(thm_dict, thm_results)
    #     for categorized_entries in paired_only.values():
    #         pair_data.append({
    #             "thm_idx": thm_idx,
    #             "state_before": categorized_entries["positive"][0]["state_before"],
    #             "positive": categorized_entries["positive"],
    #             "negative": categorized_entries["negative"],
    #         })

    # print(f"Extracted RM data with pairs from {len(pair_data)} states")
    # with open(repo_root() / cfg.rm_data_file, "w") as f:
    #     json.dump(pair_data, f, indent=4)
    #     print(f"RM data written to {cfg.rm_data_file}")
        
