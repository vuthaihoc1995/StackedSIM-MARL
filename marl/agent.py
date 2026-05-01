import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, N, q_bits):
        super().__init__()
        self.N = N
        self.q_bits = q_bits
        self.levels = 2 ** q_bits

        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, N * self.levels)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits.view(self.N, self.levels)

    def sample_action(self, logits):
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx).sum()
        return action_idx, log_prob