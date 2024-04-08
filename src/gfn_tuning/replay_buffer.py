import editdistance
import gzip
import heapq
import numpy as np
import pickle
import torch


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

    def sample(self, batch_size, theorem_id) -> tuple[torch.Tensor, torch.Tensor]:
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor

        returns (state_tactic_tensor, state_lengths, log_r)
        """
        if theorem_id not in self._buffer:
            return None, None, None
        theorem_buffer = self._buffer[theorem_id]["proofs"]
        idx = np.random.choice(
            len(theorem_buffer),
            batch_size,
            replace=True,
        )
        state_tactic_tensor = torch.nn.utils.rnn.pad_sequence(
            [theorem_buffer[i][2] for i in idx],
            batch_first=True,
            padding_value=self.termination_token_id,
        )
        state_length_tensor = torch.nn.utils.rnn.pad_sequence(
            [theorem_buffer[i][3] for i in idx],
            batch_first=True,
            padding_value=0
        )
        log_r_list = [theorem_buffer[i][0] for i in idx]
        return state_tactic_tensor, state_length_tensor, log_r_list

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