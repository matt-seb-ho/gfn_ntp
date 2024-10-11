from dataclasses import dataclass
from typing import Optional

from proof_flow.src.search.common import ProofSearchParams
from proof_flow.src.utils import CUSTOM_DEBUG_LEVEL
from proof_flow.src.prompts import REPROVER_TACGEN_WITH_HISTORY

@dataclass
class NTPConfig:
    # main parameters
    lr: float = 1e-4
    n_samples: int | list[int] = 4
    replay_batch_size: int = 4
    use_4bit: bool = False
    use_buffer_prob: float = 0.5
    repeats_per_accumulated_batch: int = 1 # for gradient accumulation
    seq2seq: bool = True
    truncate_state: bool = True
    conditional_log_z: bool = True
    branch_only_at_root: bool = True
    
    # temperature parameters
    pf_temp_prob: float = 0.666
    pf_temp_high: float = 1.0
    pf_temp_low: float = 0.25
    reward_temp_start: float = 1.0
    reward_temp_end: float = 1.0
    reward_temp_horizon: int = 750

    # constraints
    max_tactics: int = 3
    min_tactic_tokens: int = 2
    max_tactic_tokens: int = 30
    max_input_length: int = 640
    dojo_timeout: int = 600 # default comes from LeanDojo

    # checkpoints
    ckpt_dest: str = "checkpoints"
    save_ckpt_on_val: bool = False

    # logging
    # debug_log_level: 
    # - default is "GFN_DEBUG"
    # - can also use "DEBUG" to also include lean dojo logs
    debug_log_level: str = CUSTOM_DEBUG_LEVEL
    log_debug_to_stdout: bool = False
    log_debug_to_file: bool = True
    debug_log_file: str = "debug.log" # relative to repo root
