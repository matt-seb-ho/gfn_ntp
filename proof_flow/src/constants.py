# GFN tuning
GFN_POLICY_ADAPTER_NAME = "gfn_policy"
REWARD_ADAPTER_NAME = "reward"
# TACTIC_DELIMITER = "\n" 
# making this so there's no chance of it being in the tactic string
TACTIC_DELIMITER = "\n<tactic_delimiter>\n" 
PROOF_COMPLETE_MESSAGE = "no goals"
LEAN_DOJO_RANDOM_DATA_PATH = "data/leandojo_benchmark_4/random/"
DEFAULT_VERIFIER_BATCH_SIZE = 1
DEFAULT_VERIFIER_ADAPTER_NAME = "verifier"
DEFAULT_GENERATOR_ADAPTER_NAME = "generator"

# from lean dojo: 
# TacticResult = Union[
#     TacticState,
#     ProofFinished,
#     LeanError,
#     TimeoutError,
#     ProofGivenUp,
# ]
LEAN_ERROR_STRING = "LeanError"
TIMEOUT_ERROR_STRING = "TimeoutError"
PROOF_GIVEN_UP_STRING = "ProofGivenUp"
TACTIC_ERROR_STRINGS = {LEAN_ERROR_STRING, TIMEOUT_ERROR_STRING, PROOF_GIVEN_UP_STRING}
