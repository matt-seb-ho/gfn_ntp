import argparse
import json
import re
from pathlib import Path
from typing import Optional
from collections import defaultdict
from tqdm import tqdm
from lean_dojo import LeanGitRepo, get_traced_repo_path


from proof_flow.src.utils import repo_root

# pattern to match:
# "[theorem_statement]:=[optional whitespace 1][optional 'by'][optional whitespace 2][proof]"
# need to supply the re.DOTALL flag to match across lines
formal_statement_pattern = re.compile(r"(.+?):=(\s*)(by)?(\s*)(.+)", re.DOTALL)

def get_formal_statement(
    thm_info: dict, 
    repo: Optional[LeanGitRepo] = None,
    thm_idx: Optional[int] = None,
    formatting_stats: Optional[dict] = None,
) -> tuple[str, bool]:
    """
    given:  theorem file, line and column number of the start and end of the theorem (including the proof).
    goal:   extract the formal statement of the theorem, where
            if the theorem and its proof are given as f"{prefix} := (by )?{proof}",
            then the formal statement is f"prefix := (by )?"
    return: formal statement (str), is_tactic_proof (bool)
    
    implementation notes:
    The most robust approach is to use a regex that allows for arbitrary whitespace between the components.
    - This is what we implement here.
    However since we start with the lines of the theorem + proof, and I have hunch that the proof is always 
    on the line after ":=" or ":= by", I am tempted to just combine lines until := is found. 
    - I can validate with formatting stats.

    """
    # step 1: get the theorem + proof substring
    if repo is None:
        repo = LeanGitRepo(thm_info["url"], thm_info["commit"])
    repo_path = get_traced_repo_path(repo)
    file_path = repo_path / thm_info["file_path"]
    with open(file_path) as f:
        file_lines = f.readlines()
    start_line = thm_info["start"][0] - 1
    end_line = thm_info["end"][0]
    txt = "".join(file_lines[start_line:end_line])

    # step 2: match the pattern
    match = formal_statement_pattern.match(txt)
    if match is None:
        raise ValueError(f"Could not match pattern in theorem {thm_info['full_name']}")
        assert False

    # step 3: extract the formal statement
    is_tactic_proof = match.group(3) is not None
    suffix = ":= by\n" if is_tactic_proof else ":=\n"
    formal_statement = f"{match.group(1)}{suffix}"
    thm_info["formal_statement"] = formal_statement
    
    # step 4: update formatting stats
    if formatting_stats is not None:
        try:
            if is_tactic_proof:
                formatting_stats["before_by"][match.group(2)].append(thm_idx)
                formatting_stats["after_by"][match.group(4)].append(thm_idx)
            else:
                key = (match.group(2) or "") + (match.group(4) or "")
                formatting_stats["no_by"][key].append(thm_idx)
        except KeyError:
            pass
    
    return formal_statement, is_tactic_proof


def add_formal_statements(
    data: dict, 
    output_file: Optional[str] = None, 
    stats_file: Optional[str] = None
) -> tuple[dict, list]:
    dummy_thm = next(iter(data.values()))
    repo = LeanGitRepo(dummy_thm["url"], dummy_thm["commit"])
    stats = {
        "before_by": defaultdict(list),
        "after_by": defaultdict(list),
        "no_by": defaultdict(list),
        "failure": [],
    }
    for thm_idx, thm in tqdm(data.items()):
        try:
            formal_statement, is_tactic_proof = get_formal_statement(
                thm, 
                repo=repo, 
                thm_idx=thm_idx,
                formatting_stats=stats,
            )
            thm["formal_statement"] = formal_statement
            thm["is_tactic_proof"] = is_tactic_proof
        except ValueError as e:
            thm["formal_statement"] = None
            thm["is_tactic_proof"] = None
            stats["failure"].append(
                (thm_idx, thm["full_name"], thm["file_path"], str(e))
            )

    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)
    if stats_file is not None:
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=4)
    return data, stats


def main():
    psr = argparse.ArgumentParser()
    psr.add_argument("--src", type=str, default="data/eval_seed_thms.json")
    psr.add_argument("--tgt", type=str, default="data/eval_seed_thms_wfs.json")
    psr.add_argument("--stat", type=str, default="data/eval_seed_thms_failed.json")
    args = psr.parse_args()

    src_data_path = repo_root() / args.src
    tgt_data_path = repo_root() / args.tgt
    stats_file = repo_root() / args.stat
    with open(src_data_path) as f:
        data = json.load(f)

    # length filter does not add idxs
    # in original pipeline, the timing code adds idxs
    # since style check is between length and timing, we need to add idxs here
    if isinstance(data, list):
        data = {i: thm for i, thm in enumerate(data)}

    data, stats = add_formal_statements(data, tgt_data_path, stats_file)
    print(f"failure count: {len(stats['failure'])}")
    
    # with open("data/eval_seed_thms.json", 'w') as json_file:
    #     json.dump(data, json_file, indent=4)
    

if __name__ == '__main__':
    main()
