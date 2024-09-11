# script for evaluating provers with tree search
import os

import hydra
from omegaconf import OmegaConf
os.environ["RAY_DEDUP_LOGS"] = "0"
import uuid
import json
import pickle
import hashlib
import argparse
from loguru import logger
from typing import List, Tuple, Optional

from proof_flow.src.search.common import set_logger
from proof_flow.src.search.proof_search import Status, DistributedProver
from proof_flow.src.utils import (
    get_config,
    prepare_environment_for_lean_dojo,
    repo_root,
)


prepare_environment_for_lean_dojo()
from lean_dojo import ( # isort: skip
    Theorem,
    LeanGitRepo,
    Pos,
    is_available_in_cache,
)


def load_data(cfg: OmegaConf) -> tuple[LeanGitRepo, list[Theorem], list[Pos]]:
    with open(cfg.paths.input_file) as f:
        data = json.load(f)
    # assume all theorems are from the same repo
    thm0 = next(iter(data.values()))
    repo = LeanGitRepo(thm0["url"], thm0["commit"])
    theorems = []
    positions = []
    for _, thm_dict in data.items():
        theorems.append(
            Theorem(repo, thm_dict["file_path"], thm_dict["full_name"])
        )
        positions.append(Pos(*thm_dict["start"]))
    return repo, theorems, positions


def evaluate(
    cfg: OmegaConf,
    save_results: bool = False,
    exp_id: Optional[str] = None,
    verbose: bool = False,
) -> float:
    set_logger(verbose)
    repo, theorems, positions = load_data(cfg)

    # Search for proofs using multiple concurrent provers.
    quantization_config = None
    if cfg.model.use_4bit:
        quantization_config = hydra.utils.instantiate(
            cfg.model.quantization_config
        )
    search_tree_dir = (
        str(repo_root() / cfg.paths.save_search_tree_dir)
        if cfg.search.save_search_trees
        else None
    )
    prover = DistributedProver(
        cfg.search.use_vllm,
        cfg.model.id,
        None, # ret_ckpt_path
        None, # indexed_corpus_path
        cfg.search.max_input_seq_len,
        cfg.search.max_output_seq_len,
        cfg.search.length_penalty,
        None, # tactic
        None, # module
        cfg.search.num_workers,
        cfg.search.num_gpus,
        cfg.search.timeout,
        cfg.search.max_expansions,
        cfg.search.num_sampled_tactics,
        cfg.search.max_new_tokens,
        save_search_tree=search_tree_dir,
        is_peft_model=cfg.model.use_peft,
        quantization_config=quantization_config,
        debug=verbose,
    )
    results = prover.search_unordered(repo, theorems, positions)

    # Calculate the result statistics.
    num_proved = 0
    num_failed = 0
    num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    logger.info(
        f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )

    if num_proved + num_failed == 0:
        pass_1 = float("nan")
    else:
        pass_1 = num_proved / (num_proved + num_failed)

    # Save the results.
    if exp_id is None:
        exp_id = str(uuid.uuid4())

    if save_results:
        pickle_path = os.path.join(
            cfg.paths.search_eval_results_dir, 
            f"{exp_id}_results.pickle",
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)
            logger.info(f"Results saved to {pickle_path}")

    return pass_1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="search_eval",
        help="config name (sans .yaml) in the config directory",
    )
    parser.add_argument(
        "--exp-id", 
        type=str, 
        help="Experiment ID used for logging."
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Set the logging level to DEBUG."
    )
    args = parser.parse_args()

    # assert args.gen_ckpt_path or args.tactic
    # assert args.num_gpus <= args.num_workers

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)

    cfg = get_config(config_name=args.config)

    pass_1 = evaluate(
        cfg,
        save_results=True,
        exp_id=args.exp_id,
        verbose=args.verbose,
    )

    logger.info(f"Pass@1: {pass_1}")


if __name__ == "__main__":
    main()
