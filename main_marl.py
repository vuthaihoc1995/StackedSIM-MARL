# ===== IMPORTS =====
from env.stacked_ris_env import StackedRISEnv
from marl.agent import Actor
from marl.critic import Critic
from marl.marl_trainer import train

import torch
import torch.optim as optim

# ===== ENVIRONMENT =====
env = StackedRISEnv(
    K=2,
    N=8,
    q_bits=2,
    alpha0=1e-3,
    alphaK=5e-3,
    beta=0.5,
    noise_var=1.0
)

# ===== Milti-AGENTS =====
actors = [Actor(obs_dim=4, N=8, q_bits=2) for _ in range(2)]
critic = Critic(state_dim=8)


# ===== OPTIMIZERS =====
actor_optim = optim.Adam(
    [p for actor in actors for p in actor.parameters()],
    lr=1e-3
)
critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

# ===== TRAINING LOOP =====
for episode in range(10000):
    train(env, actors, critic, actor_optim, critic_optim)

    if episode % 100 == 0:
        print(f"Episode {episode} completed")
