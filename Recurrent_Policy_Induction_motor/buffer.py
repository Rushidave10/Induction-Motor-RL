from gym import spaces
import torch
import numpy as np


class Buffer:

    def __init__(self,
                 config: dict,
                 observation_space: spaces.Box,
                 action_space: spaces.Box,
                 ) -> None:

        self.samples_flat = None
        self.worker_steps = config["worker_steps"]
        self.n_mini_batch = config["n_mini_batch"]
        self.batch_size = self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        hidden_state_size = config["recurrence"]["hidden_state_size"]
        self.layer_type = config["recurrence"]["layer_type"]
        self.sequence_len = config["recurrence"]["sequence_len"]
        self.true_sequence_len = 0

        self.rewards = np.zeros(self.worker_steps, dtype=np.float32)
        self.actions = torch.zeros((self.worker_steps, action_space.shape[0]), dtype=torch.long)
        self.dones = np.zeros(self.worker_steps, dtype=np.bool)
        self.obs = torch.zeros((self.worker_steps,) + observation_space.shape)
        self.hxs = torch.zeros((self.worker_steps, hidden_state_size))
        self.cxs = torch.zeros((self.worker_steps, hidden_state_size))
        self.log_probs = torch.zeros(self.worker_steps)
        self.values = torch.zeros(self.worker_steps)
        self.advantages = torch.zeros(self.worker_steps)

    def prepare_batch_dict(self) -> None:

        samples = dict(actions=self.actions,
                       values=self.values,
                       log_probs=self.log_probs,
                       advantages=self.advantages,
                       obs=self.obs,
                       loss_mask=torch.ones(self.worker_steps, dtype=torch.float32)
                       )

        samples["hxs"] = self.hxs
        if self.layer_type == "lstm":
            samples["cxs"] = self.cxs

        episode_done_idx = []
        episode_done_idx = list(self.dones.nonzero()[0])

        if len(episode_done_idx) == 0 or episode_done_idx[-1] != self.worker_steps - 1:
            episode_done_idx.append(self.worker_steps - 1)

        max_sequence_len = 1
        for key, value in samples.items():
            sequences = []
            start_idx = 0
            for done_idx in episode_done_idx:
                episode = value[start_idx: done_idx + 1]
                start_idx = done_idx + 1
                if self.sequence_len > 0:
                    for start in range(0, len(episode), self.sequence_len):
                        end = start + self.sequence_len
                        sequences.append(episode[start:end])
                    max_sequence_len = self.sequence_len

            for i, sequence in enumerate(sequences):
                sequences[i] = self.pad_sequence(sequence, max_sequence_len)

            samples[key] = torch.stack(sequences, dim=0)

            if key == "hxs" or key == "cxs":
                samples[key] = samples[key][:, 0]

        self.true_sequence_len = max_sequence_len
        self.samples_flat = {}
        for key, value in samples.items():
            if not key == "hxs" and not key == "cxs":
                value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            self.samples_flat[key] = value

    @staticmethod
    def pad_sequence(sequence: torch.Tensor, target_length: int) -> torch.Tensor:

        delta_length = target_length - len(sequence)

        if delta_length <= 0:
            return sequence
        if len(sequence.shape) > 1:
            padding = torch.zeros(((delta_length,) + sequence.shape[1:]), dtype=sequence.dtype)
        else:
            padding = torch.zeros(delta_length, dtype=sequence.dtype)
        #  concatenate the zeros to sequence
        return torch.cat((sequence, padding), dim=0)

    def recurrent_mini_batch_generator(self) -> dict:

        num_sequences = len(self.samples_flat["values"]) // self.true_sequence_len
        num_sequences_per_batch = num_sequences // self.n_mini_batch

        num_sequences_per_batch = [num_sequences_per_batch] * self.n_mini_batch
        remainder = num_sequences % self.n_mini_batch
        for i in range(remainder):
            num_sequences_per_batch[i] += 1
        indices = torch.arange(0, num_sequences * self.true_sequence_len).reshape(num_sequences, self.true_sequence_len)
        sequence_indices = torch.randperm(num_sequences)

        start = 0
        for n_sequence in num_sequences_per_batch:
            end = start + n_sequence
            mini_batch_indices = indices[sequence_indices[start:end]].reshape(-1)
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key != "hxs" and key != "cxs":
                    mini_batch[key] = value[mini_batch_indices]
                else:
                    #  Collect only the recurrent cell states that are at beginning of the sequence
                    mini_batch[key] = value[sequence_indices[start:end]]
            start = end
            yield mini_batch

    def calc_advantages(self, last_value: torch.tensor, gamma: float, lamda: float):

        with torch.no_grad():
            last_advantage = 0
            mask = torch.tensor(self.dones).logical_not()   # mask values on terminal states
            rewards = torch.tensor(self.rewards)
            for t in reversed(range(self.worker_steps)):
                last_value = last_value * mask[t]
                last_advantage = last_advantage * mask[t]
                delta = rewards[t] + gamma * last_value - self.values[t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.advantages[t] = last_advantage
                last_value = self.values[t]
