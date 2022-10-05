import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box
from torch.distributions import MultivariateNormal


class ActorCriticModel(nn.Module):
    def __init__(self, config: dict, observation_space: Box, action_space: Box) -> None:
        super(ActorCriticModel, self).__init__()

        self.hidden_size = config["hidden_layer_size"]
        self.recurrence = config["recurrence"]
        self.training = False
        in_feature_size = observation_space.shape[0]
        action_std = config["action_std"]
        var = torch.full((action_space.shape[0],), action_std * action_std)
        self.cov_mat = torch.diag(var).unsqueeze(dim=0)

        if self.recurrence["layer_type"] == "gru":
            self.recurrent_layer = nn.GRU(input_size=in_feature_size,
                                          hidden_size=self.recurrence["hidden_state_size"],
                                          batch_first=True,
                                          )
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_layer = nn.LSTM(input_size=in_feature_size,
                                           hidden_size=self.recurrence["hidden_state_size"],
                                           batch_first=True,
                                           )

        #  Initialize recurrent layer
        for name, param in self.recurrent_layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

        # one-lstm layer combined with one MLP layers before decoupling the actor and critic network.
        self.lin_hidden = nn.Linear(self.recurrence["hidden_state_size"], self.hidden_size)  # 64, 128
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        #  Decoupling of actor and critic network
        #  Hidden_layer of policy
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)  # 128, 128
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        #  Hidden_layer of critic
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)  # 128,128
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        #  Output layer of actor
        self.policy = nn.Linear(self.hidden_size, action_space.shape[0])  # 128,2
        nn.init.orthogonal_(self.policy.weight, np.sqrt(0.01))

        #  Output layer of critic
        self.value = nn.Linear(self.hidden_size, 1)  # 128,1
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, obs: torch.Tensor, recurrent_cell: torch.Tensor, sequence_len: int = 1):

        h = obs

        if sequence_len == 1:
            h = h.unsqueeze(0)
            h, recurrent_cell = self.recurrent_layer(h.unsqueeze(1), recurrent_cell)
            h = h.squeeze(1)

        else:
            h_shape = tuple(h.size())
            h = h.reshape((h_shape[0] // sequence_len), sequence_len, h_shape[1])
            h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)
            h_shape = tuple(h.size())
            h = h.reshape(h_shape[0] * h_shape[1], h_shape[2])

        h = torch.tanh(self.lin_hidden(h))
        h_policy = torch.tanh(self.lin_policy(h))
        h_value = torch.tanh(self.lin_value(h))

        value = self.value(h_value).reshape(-1)
        if self.training:
            mu = torch.tanh(self.policy(h_policy))
            cov_mat = self.cov_mat
            pi = MultivariateNormal(mu, cov_mat)
        else:
            pi = torch.tanh(self.policy(h_policy))
        return pi, value, recurrent_cell

    def init_recurrent_cell_states(self, num_sequences: int) -> tuple:
        hxs = torch.zeros(num_sequences, self.recurrence["hidden_state_size"],
                          dtype=torch.float32).unsqueeze(0)
        cxs = None
        if self.recurrence["layer_type"] == "lstm":
            cxs = torch.zeros(num_sequences, self.recurrence["hidden_state_size"],
                              dtype=torch.float32).unsqueeze(0)
        return hxs, cxs



