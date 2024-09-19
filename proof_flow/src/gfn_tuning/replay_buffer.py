import gzip
import heapq
import pickle
from collections import deque
from typing import Optional

import editdistance
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from proof_flow.src.constants import TACTIC_DELIMITER
from proof_flow.src.gfn_tuning.proof_tree import ProofTreeNode


BUFFER_ENTRY_KEYS = ["log_r", "proof", "state_tactic_tokens", "prompt_lengths", "states"]
BUFFER_ENTRY_KEY_IDXS = {key: idx for idx, key in enumerate(BUFFER_ENTRY_KEYS)}


class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """
    def __init__(
        self, 
        buffer_size, 
        termination_token_id, 
        pad_token_id,
        sim_tolerance=0.1,
    ):
        self.buffer_size = buffer_size
        self.termination_token_id = termination_token_id
        self.pad_token_id = pad_token_id
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
            existing_proof_tokens = self._get_tactic_tokens_list(
                stt=self._buffer_item_get(buffer_item, "state_tactic_tokens"),
                pl=self._buffer_item_get(buffer_item, "prompt_lengths"),
            )
            if (
                editdistance.eval(proof_tokens, existing_proof_tokens)
                < (
                    (len(proof_tokens) + len(existing_proof_tokens)) 
                    * self.sim_tolerance
                )
            ):
                if self._buffer_item_get(buffer_item, "log_r") >= item["log_r"]:
                    return
                else:
                    self._buffer[theorem_id]["exists"].remove(
                        self._buffer_item_get(buffer_item, "proof")
                    )
                    self._buffer[theorem_id]["proofs"].remove(buffer_item)
                    heapq.heapify(self._buffer[theorem_id]["proofs"])
                    self._buffer[theorem_id]["exists"].add(item["proof"])
                    heapq.heappush(
                        self._buffer[theorem_id]["proofs"],
                        self._create_buffer_tuple(item),
                    )
                    return
        self._buffer[theorem_id]["exists"].add(item["proof"])
        if len(self._buffer[theorem_id]["proofs"]) >= self.buffer_size:
            popped = heapq.heappushpop(
                self._buffer[theorem_id]["proofs"],
                self._create_buffer_tuple(item),
            )
            self._buffer[theorem_id]["exists"].remove(popped[1])
        else:
            heapq.heappush(
                self._buffer[theorem_id]["proofs"],
                self._create_buffer_tuple(item),
            )


    def add_batch(self, theorem_id: str, items: list[dict]) -> None:
        if theorem_id not in self._buffer:
            self._buffer[theorem_id] = {
                "proofs": [],
                "exists": set(),
            }
        for item in items:
            self.add(item)


    def sample(
        self, 
        theorem_id: str, 
        batch_size: int,
        dict_format: bool = True,
    ) -> list[tuple]:
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
        if dict_format:
            return [self._recover_buffer_dict(item) for item in selected_items]
        return selected_items


    def sample_tree(self, theorem_id: str, batch_size: int) -> ProofTreeNode:
        """
        uniformly sample a batch of items from the buffer,
        and return a reconstructed proof tree containing the sampled items
        """
        sampled_items = self.sample(theorem_id, batch_size, dict_format=False)
        if sampled_items is None:
            return None
        return self._build_replay_tree(sampled_items)


    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["proofs"]:
                print(item[1])
            print("")


    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)
    

    def _get_tactic_tokens_list(self, item=None, stt=None, pl=None) -> list[int]:
        token_list = []
        iterator = zip(
            stt or item["state_tactic_tokens"], 
            pl or item["prompt_lengths"]
        ) 
        for state_tactic_tensor, state_length in iterator:
            token_list.extend([
                token_id
                for token_id in state_tactic_tensor[state_length:].tolist()
                if token_id != self.pad_token_id
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
                    (log_r, _, stt, pl, s) = buffer_selection[idx]
                    tactic = tactics[idx][depth]
                    if tactic in seen_tactics:
                        seen_tactics[tactic][1].append(idx)
                    else:
                        child = ProofTreeNode(
                            state_str=s[depth],
                            tactic=tactic,
                            depth=depth,
                            parent=node,
                            parent_tactic_tokens=stt[depth],
                            log_r=log_r,
                            prompt_length=pl[depth],
                        )
                        seen_tactics[tactic] = (child, [idx])
                for tactic, (child, selected_idxs) in seen_tactics.items():
                    if node.children is None:
                        node.children = []
                    node.children.append(child)
                    queue.append((child, selected_idxs))
                
                # token tensors may be of different lengths
                node.children_tactic_tokens = pad_sequence(
                    [child.parent_tactic_tokens for child in node.children],
                    batch_first=True,
                    padding_value=self.pad_token_id,
                )
            depth += 1
        return root


    def _create_buffer_tuple(self, item: dict) -> tuple:
        return tuple(item[key] for key in BUFFER_ENTRY_KEYS)


    def _recover_buffer_dict(self, buffer_tuple: tuple) -> dict:
        item = {}
        for key, idx in BUFFER_ENTRY_KEY_IDXS.items():
            item[key] = buffer_tuple[idx]
        return item
    

    def _buffer_item_get(self, item: tuple, key: str):
        return item[BUFFER_ENTRY_KEY_IDXS[key]]
