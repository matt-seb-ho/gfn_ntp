# proof tree data structures
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch

from proof_flow.src.constants import PROOF_COMPLETE_MESSAGE, TACTIC_DELIMITER
from proof_flow.src.utils import prepare_environment_for_lean_dojo

prepare_environment_for_lean_dojo()
from lean_dojo import ProofFinished, TacticResult, TacticState # isort: skip


@dataclass
class ProofTreeNode:
    # remaining goals
    state: Optional[TacticResult] = None
    state_str: Optional[str] = None

    # tactic applied to get to this node
    # - "" for root node
    tactic: str = ""
    
    # depth of this node
    depth: int = 0

    # children
    # - None for leaf nodes
    parent: Optional["ProofTreeNode"] = None
    children: Optional[list["ProofTreeNode"]] = None
    
    # trajectory replay info
    # - tensor (ndim=1) of parent (state, tactic) token ids
    parent_tactic_tokens: Optional[torch.Tensor] = None
    # - tensor (ndim=2) of (this state, next tactic) token ids
    children_tactic_tokens: Optional[torch.Tensor] = None
    prompt_length: Optional[int] = None
    
    
    # terms for computing loss
    log_r: Optional[float] = None
    # parent tactic's log forward probability
    tactic_logpf: Optional[torch.Tensor] = None

    # log_pf: Optional[Tensor] = None
    # log_pterm: Optional[Tensor] = None
    # log_r: Optional[Tensor] = None
    # log_r_unpenalized: Optional[Tensor] = None


    def get_trajectory_logpf(self):
        q = deque()
        node = self
        while node.tactic_logpf is not None:
            q.appendleft(node.tactic_logpf)
            node = node.parent
        return torch.cat(q)



def extract_trajectories(root: ProofTreeNode, theorem_id: str) -> list:
    """Separate the tree into a list of trajectories for the replay buffer"""
    assert root.tactic == "", "Only the root node can be separated into trajectories"
    trajectories = []
    stack = [(root, False)]
    states: list[str] = []
    tactics: list[str] = []
    parent_tactic_tokens: list[torch.Tensor] = []
    prompt_lengths: list[int] = []

    # dfs traversal
    while stack:
        node, visited = stack.pop()
        if visited:
            # backtracking
            states.pop()
            tactics.pop()
            parent_tactic_tokens.pop()
            prompt_lengths.pop()
        else:
            stack.append((node, True))
            
        # track the current trajectory
        states.append(convert_tactic_result_to_state_string(node.state))
        tactics.append(node.tactic.strip())
        parent_tactic_tokens.append(node.parent_tactic_tokens)
        prompt_lengths.append(node.prompt_length)

        if node.children:
            for child in reversed(node.children):
                stack.append((child, False))
        else:
            # we make copies for the trajectories so the backtracking doesn't affect them
            trajectories.append({
                "theorem_id": theorem_id,
                "states": states.copy(),
                # root node has an empty string tactic, so we skip it
                "tactics": tactics[1:],
                "proof": TACTIC_DELIMITER.join(tactics[1:]),
                "state_tactic_tokens": parent_tactic_tokens[1:], # consider deepcopy...
                "prompt_lengths": prompt_lengths.copy(),
                "log_r": node.log_r,
            })
    
    return trajectories


def convert_tactic_result_to_state_string(res: TacticResult) -> str:
    if isinstance(res, TacticState):
        return res.pp
    elif isinstance(res, ProofFinished):
        return PROOF_COMPLETE_MESSAGE
    else:
        # remaining TacticResult classes: LeanError, TimeoutError, ProofGivenUp
        # LeanError and ProofGivenUp have a `error` attribute
        return getattr(res, "error", "Timeout")
