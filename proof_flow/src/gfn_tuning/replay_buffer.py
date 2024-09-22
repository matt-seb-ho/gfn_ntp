import gzip
import heapq
import pickle
from collections import deque, namedtuple
from functools import cache
from typing import Optional

import editdistance
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from proof_flow.src.constants import TACTIC_DELIMITER
from proof_flow.src.gfn_tuning.proof_tree import ProofTreeNode
from proof_flow.src.utils import prepare_environment_for_lean_dojo

prepare_environment_for_lean_dojo()
from lean_dojo import LeanGitRepo, Theorem


BUFFER_ENTRY_KEYS = ["log_r", "proof", "states"]
BUFFER_ENTRY_KEY_IDXS = {key: idx for idx, key in enumerate(BUFFER_ENTRY_KEYS)}
BufferEntry = namedtuple("BufferEntry", BUFFER_ENTRY_KEYS)


class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """
    def __init__(
        self, 
        buffer_size: int, 
        termination_token_id: int, 
        pad_token_id: int,
        tokenizer: AutoTokenizer,
        sim_tolerance: float = 0.1,
    ):
        self.buffer_size = buffer_size
        self.termination_token_id = termination_token_id
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.sim_tolerance = sim_tolerance
        self.reset()


    def reset(self) -> None:
        self._buffer = {}


    def add(self, theorem_id: str, item: BufferEntry) -> None:
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # if item is already in the buffer, skip it
        if item.proof in self._buffer[theorem_id]["exists"]:
            return
        # if the edit distance between item and any item in the buffer is small, skip it
        proof_tokens = self.tokenizer.encode(item.proof)
        for buffer_item in self._buffer[theorem_id]["proofs"]:
            existing_proof_tokens = self.tokenizer.encode(buffer_item.proof)
            if (
                editdistance.eval(proof_tokens, existing_proof_tokens)
                < (
                    (len(proof_tokens) + len(existing_proof_tokens)) 
                    * self.sim_tolerance
                )
            ):
                if buffer_item.log_r >= item.log_r:
                    return
                else:
                    self._buffer[theorem_id]["exists"].remove(buffer_item.proof)
                    self._buffer[theorem_id]["proofs"].remove(buffer_item)
                    heapq.heapify(self._buffer[theorem_id]["proofs"])
                    self._buffer[theorem_id]["exists"].add(item.proof)
                    heapq.heappush(
                        self._buffer[theorem_id]["proofs"],
                        item,
                    )
                    return
        self._buffer[theorem_id]["exists"].add(item.proof)
        if len(self._buffer[theorem_id]["proofs"]) >= self.buffer_size:
            popped = heapq.heappushpop(
                self._buffer[theorem_id]["proofs"],
                item,
            )
            self._buffer[theorem_id]["exists"].remove(popped.proof)
        else:
            heapq.heappush(
                self._buffer[theorem_id]["proofs"],
                item,
            )


    def add_batch(self, theorem_id: str, items: list[dict]) -> None:
        if theorem_id not in self._buffer:
            self._buffer[theorem_id] = {
                "proofs": [],
                "exists": set(),
            }
        for item in items:
            self.add(theorem_id, item)


    def sample(
        self, 
        theorem_id: str, 
        batch_size: int,
    ) -> list[BufferEntry]:
        """
        uniformly sample a batch of items from the buffer,
        and returns a list of n proof trajectories 
        """
        if theorem_id not in self._buffer:
            return None
        theorem_buffer = self._buffer[theorem_id]["proofs"]
        if len(theorem_buffer) == 0:
            return None
        elif len(theorem_buffer) > batch_size:
            idxs = np.random.choice(
                len(theorem_buffer),
                batch_size,
                # replace=True,
            )
            selected_items = [theorem_buffer[idx] for idx in idxs]
        else:
            selected_items = theorem_buffer
        return selected_items


    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["proofs"]:
                print(item[1])
            print("")


    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)


def extract_ground_truth_trajectory(thm_dict: dict) -> tuple[str, BufferEntry]:
    states = []
    tactics = []
    for tt in thm_dict["traced_tactics"]:
        states.append(tt["state_before"])
        tactics.append(tt["tactic"])
    states.append(thm_dict["traced_tactics"][-1]["state_after"])

    thm_uid = _get_thm_uid_from_dict(thm_dict)   
    return thm_uid, BufferEntry(
        log_r=0,
        proof=TACTIC_DELIMITER.join(tactics),
        states=states,
    )


def _get_thm_uid_from_dict(thm_dict: dict) -> str:
    repo = _get_lean_git_repo(thm_dict["url"], thm_dict["commit"])
    thm = Theorem(repo, thm_dict["file_path"], thm_dict["full_name"])
    return thm.uid


@cache
def _get_lean_git_repo(url, commit):
    return LeanGitRepo(url, commit)
