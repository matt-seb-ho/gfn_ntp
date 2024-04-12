import gzip
import heapq
import pickle
from collections import deque
from typing import Optional

import editdistance
import numpy as np
import torch
from proof_tree import ProofTreeNode

from constants import TACTIC_DELIMITER


class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """

    def __init__(self, buffer_size, termination_token_id, sim_tolerance=0.25):
        self.buffer_size = buffer_size
        self.termination_token_id = termination_token_id
        self.sim_tolerance = sim_tolerance
        self.reset()

    def reset(self) -> None:
        self._buffer = {}

    def add(self, item: dict) -> None:
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # if item is already in the buffer, skip it
        theorem_id = item["theorem_id"]
        if item["proof"] in self._buffer[theorem_id]["exists"]:
            return
        # if the edit distance between item and any item in the buffer is small, skip it
        proof_tokens = self._get_tactic_tokens_list(item=item)
        for buffer_item in self._buffer[theorem_id]["proofs"]:
            existing_proof_tokens = self._get_tactic_tokens_list(stt=buffer_item[2], sl=buffer_item[3])
            if (
                editdistance.eval(proof_tokens, existing_proof_tokens)
                < (len(proof_tokens) + len(existing_proof_tokens)) * self.sim_tolerance
            ):
                if buffer_item[0] >= item["log_r"]:
                    return
                else:
                    self._buffer[theorem_id]["exists"].remove(buffer_item[1])
                    self._buffer[theorem_id]["proofs"].remove(buffer_item)
                    heapq.heapify(self._buffer[theorem_id]["proofs"])
                    self._buffer[theorem_id]["exists"].add(item["str_tactics"])
                    heapq.heappush(
                        self._buffer[theorem_id]["proofs"],
                        (
                            item["log_r"],
                            item["proof"],
                            item["state_tactic_tensors"],
                            item["state_lengths"],
                            item["states"],
                        ),
                    )
                    return
        self._buffer[theorem_id]["exists"].add(item["proof"])
        if len(self._buffer[theorem_id]["proofs"]) >= self.buffer_size:
            popped = heapq.heappushpop(
                self._buffer[theorem_id]["proofs"],
                (
                    item["log_r"],
                    item["proof"],
                    item["state_tactic_tensors"],
                    item["state_lengths"],
                    item["states"],
                ),
            )
            self._buffer[theorem_id]["exists"].remove(popped[1])
        else:
            heapq.heappush(
                self._buffer[theorem_id]["sentences"],
                (
                    item["log_r"],
                    item["proof"],
                    item["state_tactic_tensors"],
                    item["state_lengths"],
                    item["states"],
                ),
            )

    def add_batch(self, items: list[dict]) -> None:
        for item in items:
            self.add(item)

    def sample(self, theorem_id: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor

        returns (state_tactic_tensor, state_lengths, log_r)
        """
        if theorem_id not in self._buffer:
            return None
        theorem_buffer = self._buffer[theorem_id]["proofs"]
        idxs = np.random.choice(
            len(theorem_buffer),
            batch_size,
            replace=True,
        )
        return self._build_replay_tree([theorem_buffer[idx] for idx in idxs])

    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["proofs"]:
                print(item[1])
            print("")

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)
    
    def _get_tactic_tokens_list(self, item=None, stt=None, sl=None) -> list[int]:
        token_list = []
        iterator = zip(
            stt or item["state_tactic_tensors"], 
            sl or item["state_lengths"]
        ) 
        for state_tactic_tensor, state_length in iterator:
            token_list.extend([
                token_id
                for token_id in state_tactic_tensor[state_length:].tolist()
                if token_id != self.termination_token_id
            ])
        return token_list
    
    def _build_replay_tree(self, buffer_selection: list) -> ProofTreeNode:
        # build ProofTreeNode from buffer_selection
        # level order traversal
        root = ProofTreeNode(state_str=buffer_selection[0][4][0])
        queue = deque([root, list(range(len(buffer_selection)))])
        tactics = [buffer_item[1].split(TACTIC_DELIMITER) for buffer_item in buffer_selection]
        depth = 1
        while queue:
            level_size = len(queue)
            for _ in range(level_size):
                node, selected_idxs = queue.popleft()
                seen_tactics = {}
                for idx in selected_idxs:
                    (log_r, _, stt, sl, s) = buffer_selection[idx]
                    tactic = tactics[idx][depth]
                    if tactic in seen_tactics:
                        seen_tactics[tactic][1].append(idx)
                    else:
                        child = ProofTreeNode(
                            state_str=s[depth],
                            tactic=tactic,
                            depth=depth,
                            parent=node,
                            token_tensor=stt[depth],
                            log_r=log_r,
                            prompt_length=sl[depth],
                        )
                        seen_tactics[tactic] = (child, [idx])
                for tactic, (child, selected_idxs) in seen_tactics.items():
                    if node.children is None:
                        node.children = []
                    node.children.append(child)
                    queue.append((child, selected_idxs))
            depth += 1
        return root
