"""
Sampling proof search trees to collect negative data
"""
import sys
sys.path.append('/home/vincentzhu/ReProver')
import os
import uuid
import json
import pickle
import argparse
from loguru import logger
from typing import Optional
from pathlib import Path

from proof_flow.scripts.common import prepare_environment_for_lean_dojo, set_logger
prepare_environment_for_lean_dojo("config.yaml")

from lean_dojo import LeanGitRepo, Theorem, Pos 
from prover.proof_search_tree import Status, DistributedProver


def handle_output_dir_path(path):
    # Check if the path exists
    if os.path.exists(path):
        # If it exists, check if it is a directory
        if os.path.isdir(path):
            logger.info(f"Validated {path} as a directory.")
        else:
            # If it exists and is not a directory, raise an error
            raise FileExistsError(f"The given output directory path {path} exists but is not a directory.")
    else:
        # If the path does not exist, create the directory
        os.makedirs(path)
        logger.info(f"Output directory {path} created.")


def read_data_file(data_path: str) -> tuple[LeanGitRepo, list[Theorem], list[Pos]]:
    # read file
    with open(data_path) as f:
        input_data = json.load(f)
    
    # handle list[thm_dict], dict[int, thm_dict] formats
    if isinstance(input_data, list):
        thm0 = input_data[0]
        thm_iter = input_data
    else:
        assert isinstance(input_data, dict)
        if "theorems" in input_data:
            input_data = input_data["theorems"]
        thm0 = next(iter(input_data.values()))
        thm_iter = input_data.values()

    # construct lean dojo objects
    repo = LeanGitRepo(thm0["url"], thm0["commit"])
    theorems = []
    positions = []
    for thm_info in thm_iter:
        theorems.append(Theorem(repo, thm_info["file_path"], thm_info["full_name"]))
        positions.append(Pos(*thm_info["start"]))

    return repo, theorems, positions


def sample_trees(
    data_path: str,
    exp_id: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    ret_ckpt_path: Optional[str] = None,
    indexed_corpus_path: Optional[str] = None,
    max_inp_seq_len: int = 2048,
    max_oup_seq_len: int = 512,
    length_penalty: float = 0.0,
    tactic: Optional[str] = None,
    module: Optional[str] = None,
    num_sampled_tactics: int = 64,
    timeout: int = 600,
    max_expansions: Optional[int] = None,
    num_cpus: int = 1,
    num_gpus: int = 0,
    verbose: bool = False,
    hf_generator_id: Optional[str] = None,
    hf_retrieval_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_theorems: Optional[int] = None,
    theorem_name: Optional[str] = None,
    log_file: Optional[str] = None,
) -> tuple[float, list[dict]]:
    set_logger(verbose, log_file)

    # create the output dir if it doesn't exist
    if output_dir:
        handle_output_dir_path(output_dir)

    # original version
    # repo, theorems, positions = _get_theorems(
    #     data_path, split, file_path, full_name, name_filter, num_theorems
    # )
    # updated to read format from filtering
    repo, theorems, positions = read_data_file(data_path)
    if theorem_name:
        for thm, pos in zip(theorems, positions):
            if thm.full_name == theorem_name:
                break
        assert thm.full_name == theorem_name, f"args.theorem_name {theorem_name} not found"
        logger.info(f"arg.theorem_name {theorem_name} found.")
        theorems = [thm]
        positions = [pos]
    elif num_theorems is not None:
        theorems = theorems[:num_theorems]
        positions = positions[:num_theorems]
    logger.info("Finished reading in theorem data; constructing prover...")
    
    # Search for proofs using multiple concurrent provers.
    # prover = DistributedProver(
    #     ckpt_path,
    #     indexed_corpus_path,
    #     tactic,
    #     module,
    #     num_cpus,
    #     num_gpus,
    #     timeout=timeout,
    #     num_sampled_tactics=num_sampled_tactics,
    #     debug=verbose,
    #     hf_generator_id=hf_generator_id,
    #     hf_retriever_id=hf_retrieval_id,
    # )
    prover = DistributedProver(
        ckpt_path,
        ret_ckpt_path,
        indexed_corpus_path,
        max_inp_seq_len,
        max_oup_seq_len,
        length_penalty,
        tactic=tactic,
        module=module,
        num_workers=num_cpus,
        num_gpus=num_gpus,
        timeout=timeout,
        max_expansions=max_expansions,
        num_sampled_tactics=num_sampled_tactics,
        debug=verbose,
    )

    logger.info("Prover constructed; starting proof search...")

    results = prover.search_unordered_and_save_trees(
        repo,
        theorems, 
        positions,
        output_dir,
    )

    # Calculate the result statistics.
    num_proved = num_failed = num_discarded = 0
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
    pickle_path = f"{exp_id}_results.pickle"
    pickle.dump(results, open(pickle_path, "wb"))
    logger.info(f"Results saved to {pickle_path}")

    # return pass_1, trees
    return pass_1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Runs proof search to sample possible proof trajectories "
            "and saves search trees to pickle files."
        )
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data extracted by LeanDojo (e.g., data/leandojo_benchmark/random).",
    )
    parser.add_argument("--exp-id", type=str, help="Experiment ID used for logging.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Checkpoint of the tactic generator.",
    )
    parser.add_argument(
        "--ret_ckpt_path",
        type=str,
        help="Checkpoint of the premise retriever.",
    )
    parser.add_argument(
        "--hf_gen_id",
        type=str,
        help="hf repo/id of the tactic generator.",
    )
    parser.add_argument(
        "--hf_ret_id",
        type=str,
        help="hf repo/id of the retriever.",
    )
    parser.add_argument(
        "--indexed-corpus-path",
        type=str,
        help="Path to a pickled indexed corpus. Not required for models w/o retrieval.",
    )
    parser.add_argument("--max-inp-seq-len", type=int, default=2048)
    parser.add_argument("--max-oup-seq-len", type=int, default=512)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--tactic", type=str, help="The tactic to evaluate.")
    parser.add_argument("--module", type=str, help="The module to import the tactic.")
    parser.add_argument(
        "--num-sampled-tactics",
        type=int,
        default=64,
        help="Number of tactics to sample at each node during proof search.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum number of seconds the proof search can take.",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=None,
        help="Maximum number of expansions during proof search.",
    )
    parser.add_argument(
        "--num-cpus", type=int, default=1, help="The number of concurrent provers."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="The number of GPUs for proof search."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Set the logging level to DEBUG."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="path to directory where trees will be serialized to"
    )
    parser.add_argument(
        "--num_theorems",
        type=int,
        help="how many theorems to run proof search for",
    )
    parser.add_argument(
        "--theorem_name",
        type=str,
        help="theorem name to test against specific theorems"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="log file to write to"
    )
    args = parser.parse_args()

    assert args.ckpt_path or args.tactic or args.hf_gen_id

    if args.log_file:
        logger.add(args.log_file)
    logger.info(f"PID: {os.getpid()}")
    logger.info(args)

    pass_1 = sample_trees(
        args.data_path,
        args.exp_id,
        args.ckpt_path,
        args.ret_ckpt_path,
        args.indexed_corpus_path,
        args.max_inp_seq_len,
        args.max_oup_seq_len,
        args.length_penalty,
        args.tactic,
        args.module,
        args.num_sampled_tactics,
        args.timeout,
        args.max_expansions,
        args.num_cpus,
        args.num_gpus,
        args.verbose,
        hf_generator_id=args.hf_gen_id,
        hf_retrieval_id=args.hf_ret_id,
        output_dir=args.output_dir,
        num_theorems=args.num_theorems,
        theorem_name=args.theorem_name,
        log_file=args.log_file,
    )
    logger.info(f"Pass@1: {pass_1}")


if __name__ == "__main__":
    main()