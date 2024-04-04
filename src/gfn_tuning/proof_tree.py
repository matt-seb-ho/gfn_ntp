# proof tree data structures
from typing import Optional
from torch import Tensor
from dataclasses import dataclass
from lean_dojo import TacticResult

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
    children: Optional[list["ProofTreeNode"]] = None
    
    # terms for computing loss
    prompt_length: int = -1
    step_log_pf: Optional[Tensor] = None
    trajectory_log_pf: Optional[list[Tensor]] = None

    # log_pf: Optional[Tensor] = None
    # log_pterm: Optional[Tensor] = None
    # log_r: Optional[Tensor] = None
    # log_r_unpenalized: Optional[Tensor] = None