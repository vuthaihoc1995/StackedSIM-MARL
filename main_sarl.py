from marl.sarl_trainer import train_sarl
from marl.agent import Actor
from marl.critic import Critic
from env.stacked_ris_env import StackedRISEnv
import torch.optim as optim

K, N, q_bits = 2, 8, 2

env = StackedRISEnv(
    K=K, N=N, q_bits=q_bits,
    alpha0=1e-3, alphaK=5e-3,
    beta=0.5, noise_var=1.0
)

actor_sarl = Actor(obs_dim=4*K, N=N*K, q_bits=q_bits)
critic_sarl = Critic(state_dim=4*K)

actor_sarl.train()
critic_sarl.train()

opt_actor = optim.Adam(actor_sarl.parameters(), lr=1e-3)
opt_critic = optim.Adam(critic_sarl.parameters(), lr=1e-3)

for episode in range(10000):
    train_sarl(env, actor_sarl, critic_sarl, opt_actor, opt_critic)

    if episode % 500 == 0:
        print(f"SARL training episode {episode}")