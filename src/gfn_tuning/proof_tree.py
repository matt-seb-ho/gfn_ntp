# proof tree data structures
import torch
from typing import Optional
from dataclasses import dataclass
from lean_dojo import TacticResult
from collections import deque

@dataclass
class ProofTreeNode:
    # remaining goals
    state: TacticResult

    # tactic applied to get to this node
    # - "" for root node
    tactic: str = ""
    
    # depth of this node
    depth: int = 0

    # children
    # - None for leaf nodes
    parent: Optional["ProofTreeNode"] = None
    children: Optional[list["ProofTreeNode"]] = None
    # tensor (ndim=2) of sampled tactic token ids
    # - useful for recomputing log_pf on replayed trajectories
    next_tactic_token_ids: Optional[torch.Tensor] = None
    
    # terms for computing loss
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