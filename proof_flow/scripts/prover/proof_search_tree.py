"""Proof search using best-first search.
"""

import os
import sys
import ray
import time
import heapq
import torch
import asyncio
import gc
import pickle
from lean_dojo import (
    Pos,
    Dojo,
    Theorem,
    LeanGitRepo,
    TacticState,
    LeanError,
    TimeoutError,
    ProofFinished,
    ProofGivenUp,
    DojoInitError,
    DojoCrashError,
    DojoTacticTimeoutError,
)
from loguru import logger
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ray.util.actor_pool import ActorPool
import traceback

from tqdm import tqdm
from proof_flow.scripts.common import zip_strict, TRACEBACK_LOG_LEVEL
from proof_flow.scripts.prover.search_tree import *
# from proof_flow.scripts.generation.model import RetrievalAugmentedGenerator, FixedTacticGenerator
from proof_flow.scripts.prover.tactic_generator import (
    TacticGenerator,
    HuggingFaceGenerator,
    # RetrievalAugmentedGenerator,
    FixedTacticGenerator,
    # VllmGenerator,
)


@dataclass(frozen=True)
class SearchResult:
    """The result of attempting to prove a theorem."""

    theorem: Theorem
    status: Status
    proof: Optional[List[str]]

    # Some statistics during proof search.
    actor_time: float
    environment_time: float
    total_time: float
    num_total_nodes: int
    num_searched_nodes: int


class BestFirstSearchProver:
    """A prover that uses best-first search to find proofs using a tactic generator."""

    def __init__(
        self,
        tac_gen,  # A given tactic generator.
        timeout: int,
        max_expansions: int,
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        self.tac_gen = tac_gen
        self.tac_gen.initialize()
        self.timeout = timeout
        self.max_expansions = max_expansions
        self.num_sampled_tactics = num_sampled_tactics
        self.debug = debug

        self.num_expansions = 0
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.total_time = None

    def search(
        self, 
        repo: LeanGitRepo, 
        thm: Theorem, 
        pos: Pos, 
        save_to_dir: Optional[str] = None,
    ) -> Optional[SearchResult]:
        logger.info(f"Proving {thm}")

        self.repo = repo
        self.theorem = thm
        self.posision = pos
        self.actor_time = 0.0
        self.environment_time = 0.0
        self.num_expansions = 0

        if isinstance(self.tac_gen, FixedTacticGenerator):
            imps = [self.tac_gen.module]
        else:
            imps = []

        try:
            with Dojo(
                thm, 
                timeout=60 + self.timeout, 
                additional_imports=imps
            ) as (dojo, init_state):
                self.dojo = dojo
                self.root = InternalNode(
                    state=init_state,
                    cumulative_logprob=0.0,
                )
                self.nodes = {init_state: self.root}
                # self.priority_queue = [self.root]

                with torch.no_grad():
                    try:
                        # self._best_first_search()
                        asyncio.run(self._best_first_search())
                    except DojoCrashError as ex:
                        logger.warning(f"Dojo crashed with {ex} when proving {thm}")
                    except Exception as ex:
                        # log error, but continue to save proof tree
                        logger.error(
                            f"exception while proving {thm}\n"
                            f"- type: {type(ex)}\n"
                            f"- str(e): {str(ex)}"
                        )
                        logger.log(TRACEBACK_LOG_LEVEL, traceback.format_exc())

            if self.root.status == Status.PROVED:
                proof = [e.tactic for e in self.root.extract_proof()]
            else:
                proof = None

            result = SearchResult(
                theorem=thm,
                status=self.root.status,
                proof=proof,
                actor_time=self.actor_time,
                environment_time=self.environment_time,
                total_time=self.total_time,
                num_total_nodes=len(self.nodes),
                num_searched_nodes=self.num_expansions,
            )
            logger.info(result)
            
            if save_to_dir:
                tree_file_path = os.path.join(save_to_dir, f"{thm.full_name}.pickle")
                with open(tree_file_path, 'wb') as f:
                    pickle.dump(self.root, f)
                    logger.info(f"Serialized search tree (self.root) to {tree_file_path}")
            
            return result

        except DojoInitError as ex:
            logger.warning(ex)
            return None

    async def _best_first_search(self) -> None:
        time_start = time.monotonic()
        
        priority_queue = asyncio.PriorityQueue()
        priority_queue.put_nowait((-self.root.priority, self.root))
        
        while True:
            # if len(self.priority_queue) == 0:
            if priority_queue.empty():
                logger.info("Ran out of nodes to search.")
                break

            try:
                # self._step()
                await self._step(priority_queue)
            except DojoTacticTimeoutError:
                assert time.monotonic() - time_start >= self.timeout

            self.total_time = time.monotonic() - time_start
            # if self.total_time > self.timeout:
            #     if self.root.status == Status.PROVED:
            #         logger.info("Found a proof but timed out.")
            #     self.root.status = Status.OPEN
            #     logger.info("Search timed out.")
            #     break
            if self.total_time > self.timeout or (
                self.max_expansions is not None
                and self.num_expansions > self.max_expansions
            ):
                if self.root.status == Status.PROVED:
                    logger.info("Found a proof!")
                self.root.status = Status.OPEN
                logger.info("Hit the resource limit (timeout or max_expansions).")
                break

            if self.root.status == Status.FAILED:
                logger.info("Failed early!")
                break

            if self.root.status == Status.PROVED:
                logger.info("Found a proof!")
                break

    async def _step(self, priority_queue):
        """
        Perform a single step of search.

        Selects the node with the highest priority, queries the model for suggested
        tactics, and tries each tactic in the environment, creating and enqueuing
        a new node for each valid result.
        """
        # Search the node with highest priority.
        # search_node = heapq.heappop(self.priority_queue)
        try:
            _, search_node = priority_queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        logger.debug(f"Expanding node: {search_node}")

        if self.debug:
            assert all(
                # search_node.priority >= node.priority for node in self.priority_queue
                search_node.priority >= node.priority for node in priority_queue
            )

        if isinstance(search_node.state, TacticState):
            ts = search_node.state.pp
        else:
            ts = search_node.state.unsolved_tactic_state
        # suggestions = self._generate_tactics(ts)
        suggestions = await self._generate_tactics(ts)

        # Try all tactics in order of descending logprob, and collect the results. Any
        # new nodes are added to `self.nodes`, and edges are added to the result node.
        results = []
        for tactic, logprob in suggestions:
            # edge, finished = self._run_tactic(search_node, tactic, logprob)
            edge, finished = self._run_tactic(search_node, tactic, logprob, priority_queue)
            results.append(edge)
            if finished:
                break

        # Store the fixed out edges of this node, marking it as explored.
        # This will trigger recursively recomputing tree statistics.
        search_node.out_edges = results
        self.num_expansions += 1
        priority_queue.task_done()
        
        # If we're running in debug mode, run a full test suite each step
        if self.debug:
            assert self.num_expansions == sum(
                node.is_explored
                for node in self.nodes.values()
                if isinstance(node, InternalNode)
            )
            self.check_invariants(priority_queue)

    @torch.no_grad()
    async def _generate_tactics(self, ts: str) -> List[Tuple[str, float]]:
        t0 = time.monotonic()

        path = str(self.theorem.file_path)

        if self.theorem.repo != self.repo:
            path = self.theorem.repo.get_packages_dir() / self.theorem.repo.name / path

        # suggestions = self.tac_gen.generate(
        suggestions = await self.tac_gen.generate(
            state=ts,
            file_path=path,
            theorem_full_name=self.theorem.full_name,
            theorem_pos=self.posision,
            num_samples=self.num_sampled_tactics,
        )

        self.actor_time += time.monotonic() - t0

        logger.debug(f"Tactic suggestions: {suggestions}")
        return suggestions

    def _run_tactic(
        self, node: InternalNode, tactic: str, logprob: float, priority_queue
    ) -> Tuple[Edge, bool]:
        t0 = time.monotonic()
        response = self.dojo.run_tac(node.state, tactic)

        elapsed = time.monotonic() - t0
        self.environment_time += elapsed

        try:
            # If we've seen this response before, use the existing node
            result_node = self.nodes[response]
        except KeyError:
            # Build a new node
            if isinstance(response, ProofFinished):
                result_node = ProofFinishedNode(response)
            elif type(response) in (
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                result_node = ErrorNode(response)
            else:
                assert isinstance(response, TacticState)
                result_node = InternalNode(
                    state=response,
                    cumulative_logprob=logprob + node.cumulative_logprob,
                )

            if result_node.status == Status.OPEN:  # Don't search proved/failed nodes
                # heapq.heappush(self.priority_queue, result_node)  # type: ignore
                priority_queue.put_nowait((-result_node.priority, result_node))

        # Record the new node and add it to the search queue.
        self.nodes[response] = result_node

        # Build an edge connecting these nodes.
        # Will be added to the source node externally.
        edge = Edge(tactic=tactic, src=node, dst=result_node)

        if isinstance(result_node, InternalNode):
            result_node.in_edges.append(edge)

        return edge, isinstance(response, ProofFinished)

    #########
    # DEBUG #
    #########

    def check_invariants(self, priority_queue):
        """Perform some sanity checks."""
        # for node in self.priority_queue:
        for node in priority_queue:
            assert node in self.nodes.values()
            assert isinstance(node, InternalNode)
            assert not node.is_explored

        for response, node in self.nodes.items():
            if isinstance(response, ProofFinished):
                assert isinstance(node, ProofFinishedNode)
                # assert node not in self.priority_queue
                assert node not in priority_queue
                assert self.root.status == Status.PROVED
            elif type(response) in (
                LeanError,
                TimeoutError,
                ProofGivenUp,
            ):
                assert isinstance(node, ErrorNode)
                # assert node not in self.priority_queue
                assert node not in priority_queue
            else:
                assert isinstance(node, InternalNode)

                if node.is_explored:
                    # assert node not in self.priority_queue
                    assert node not in priority_queue
                else:
                    # assert node in self.priority_queue
                    assert node not in priority_queue

                node.check_invariants(priority_queue)


@ray.remote
class ProverActor:
    """Ray actor for running an instance of `BestFirstSearchProver`."""

    def __init__(
        self,
        tac_gen: TacticGenerator,
        timeout: int,
        max_expansions: int,
        num_sampled_tactics: int,
        debug: bool,
    ) -> None:
        self.prover = BestFirstSearchProver(
            tac_gen,
            timeout,
            max_expansions,
            num_sampled_tactics,
            debug,
        )

    def search(
        self, repo: LeanGitRepo, thm: Theorem, pos: Pos, save_to_dir: Optional[str] = None
    ) -> Optional[SearchResult]:
        return self.prover.search(repo, thm, pos, save_to_dir)


class DistributedProver:
    """A distributed prover that uses Ray to parallelize the proof search.

    It is a wrapper around `CpuProver` and `GpuProver` that handles the different
    devices and different number of concurrent provers.
    """

    def __init__(
        self,
        ckpt_path: Optional[str],
        ret_ckpt_path: Optional[str],
        indexed_corpus_path: Optional[str],
        max_inp_seq_len: int,
        max_oup_seq_len: int,
        length_penalty: float,
        tactic: Optional[str],
        module: Optional[str],
        num_workers: int,
        num_gpus: int,
        max_expansions: int,
        timeout: int,
        num_sampled_tactics: int,
        debug: Optional[bool] = False,
    ) -> None:
        if ckpt_path is None:
            assert tactic and not indexed_corpus_path
        else:
            assert not tactic and not module

        if ckpt_path is None:
            tac_gen = FixedTacticGenerator(tactic, module)
        else:
            device = torch.device("cuda") if num_gpus > 0 else torch.device("cpu")
            tac_gen = HuggingFaceGenerator(
                ckpt_path, device, max_inp_seq_len, max_oup_seq_len, length_penalty
            )

        self.distributed = num_workers > 1
        if not self.distributed:
            assert num_gpus <= 1
            self.prover = BestFirstSearchProver(
                tac_gen, timeout, num_sampled_tactics, debug
            )
            return

        if num_gpus >= 1:
            logger.info(f"Launching {num_workers} workers with {num_gpus} GPUs.")
            num_gpus_per_worker = num_gpus / num_workers
            provers = [
                ProverActor.options(num_gpus=num_gpus_per_worker).remote(
                    tac_gen,
                    timeout=timeout,
                    max_expansions=max_expansions,
                    num_sampled_tactics=num_sampled_tactics,
                    debug=debug,
                )
                for _ in range(num_workers)
            ]
        else:
            logger.info(f"Launching {num_workers} CPU workers.")
            provers = [
                ProverActor.remote(
                    tac_gen,
                    timeout=timeout,
                    max_expansions=max_expansions,
                    num_sampled_tactics=num_sampled_tactics,
                    debug=debug,
                )
                for _ in range(num_workers)
            ]

        self.prover_pool = ActorPool(provers)

    def search_unordered(
        self, repo: LeanGitRepo, theorems: List[Theorem], positions: List[Pos]
    ) -> List[Optional[SearchResult]]:
        """Parallel proof search for `theorems`. The order of the results is not guaranteed to match the order of the input."""
        if not self.distributed:
            return [
                self.prover.search(repo, thm, pos)
                for thm, pos in zip_strict(theorems, positions)
            ]

        try:
            results = list(
                tqdm(
                    self.prover_pool.map_unordered(
                        lambda p, x: p.search.remote(repo, x[0], x[1]),
                        zip_strict(theorems, positions),
                    ),
                    total=len(theorems),
                    desc="Searching theorems"
                )
            )
        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)

        return results

    def search_unordered_and_save_trees(
        self, 
        repo: LeanGitRepo,
        theorems: List[Theorem],
        positions: List[Pos],
        save_to_dir: str,
    ) -> List[SearchResult]:
        """Parallel proof search for `theorems`. The order of the results is not guaranteed to match the order of the input."""
        if not self.distributed:
            logger.info("Running search_unordered_and_save_trees in non-distributed mode")
            results = []
            for thm, pos in tqdm(zip_strict(theorems, positions), total=len(theorems), desc="Searching theorems"):
                try:
                    res = self.prover.search(
                        repo, 
                        thm, 
                        pos, 
                        save_to_dir=save_to_dir
                    )
                    results.append(res)
                except Exception as e:
                    logger.error(f"exception occurred in search_unordered_...: {type(e)}, {e}")
                    logger.log(TRACEBACK_LOG_LEVEL, traceback.format_exc())

            return results

        def actor_pool_search(
            p,  
            x,
        ):
            try:
                result = p.search.remote(
                    repo, 
                    x[0], 
                    x[1], 
                    save_to_dir=save_to_dir
                )
                return result
            
            except RuntimeError as e:
                msg = str(e)
                if "CUDA" in msg and "out of memory" in msg:
                    thm_name = x[0].full_name
                    logger.info(f"Caught a CUDA OOM while proving {thm_name}, continuing to next example")
                    # Attempt to clear cache and recover here
                    gc.collect()
                    torch.cuda.empty_cache()
                    return None
                else:
                    logger.error(f"Encountered error: {msg}, returning None and continuing")
                    return None
            
        try:
            results = list(
                tqdm(
                    self.prover_pool.map_unordered(
                        lambda p, x: actor_pool_search(p, x),
                        zip_strict(theorems, positions),
                    ),
                    total=len(theorems),
                    desc="Searching theorems"
                )
            )
            
            filtered_results = [res for res in results if res is not None]

        except ray.exceptions.RayActorError as ex:
            logger.error(ex)
            sys.exit(1)

        return filtered_results
    