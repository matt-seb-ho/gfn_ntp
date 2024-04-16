# proof tree data structures
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
from lean_dojo import (
    LeanError,
    ProofFinished,
    ProofGivenUp,
    TacticResult,
    TacticState,
    TimeoutError
)

from src.constants import PROOF_COMPLETE_MESSAGE, TACTIC_DELIMITER


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
    
    # tensor (ndim=1) of previous state and action (for replay buffer)
    token_tensor: Optional[torch.Tensor] = None
    # tensor (ndim=2) of sampled tactic token ids
    next_tactic_token_ids: Optional[torch.Tensor] = None
    
    # terms for computing loss
    log_r: Optional[float] = None
    prompt_length: int = -1
    tactic_logpf: Optional[float] = None
    trajectory_logpf: Optional[torch.Tensor] = None

    # log_pf: Optional[Tensor] = None
    # log_pterm: Optional[Tensor] = None
    # log_r: Optional[Tensor] = None
    # log_r_unpenalized: Optional[Tensor] = None

    def get_trajectory_logpf(self):
        if self.trajectory_logpf is not None:
            return self.trajectory_logpf
        q = deque()
        node = self
        while node.tactic_logpf is not None:
            q.appendleft(node.tactic_logpf)
            node = node.parent
        self.trajectory_logpf = torch.cat(q)
        return self.trajectory_logpf


def extract_trajectories(root: ProofTreeNode, theorem_id: str) -> list:
    """Separate the tree into a list of trajectories for the replay buffer"""
    assert root.tactic == "", "Only the root node can be separated into trajectories"
    trajectories = []
    stack = [(root, False)]
    states: list[str] = []
    tactics: list[str] = []
    state_tactic_tensors: list[torch.Tensor] = []
    state_lengths: list[int] = []

    # dfs traversal
    while stack:
        node, visited = stack.popleft()
        if visited:
            # backtracking
            states.pop()
            tactics.pop()
            state_tactic_tensors.pop()
            state_lengths.pop()
        else:
            stack.append((node, True))
            
        # track the current trajectory
        states.append(convert_tactic_result_to_state_string(node.state))
        tactics.append(node.tactic.strip())
        state_tactic_tensors.append(node.token_tensor)
        state_lengths.append(node.prompt_length)

        if node.children:
            for child in reversed(node.children):
                stack.append((child, False))
        else:
            trajectories.append({
                "theorem_id": theorem_id,
                "states": states.copy(),
                "tactics": tactics.copy(),
                "proof": TACTIC_DELIMITER.join(tactics[1:]),
                "state_tactic_tensors": deepcopy(state_tactic_tensors),
                "state_lengths": state_lengths.copy(),
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
