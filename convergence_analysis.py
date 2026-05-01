import torch
import numpy as np
import matplotlib.pyplot as plt

from env.stacked_ris_env import StackedRISEnv
from marl.agent import Actor
from marl.critic import Critic
from marl.marl_trainer import train as train_marl
from marl.sarl_trainer import train_sarl


# =====================================================
# System setup (same as training)
# =====================================================
K, N, q_bits = 2, 8, 2
EPISODES = 5000
MA_WINDOW = 100

env_marl = StackedRISEnv(
    K, N, q_bits,
    alpha0=1e-3, alphaK=5e-3,
    beta=0.5, noise_var=1.0
)

env_sarl = StackedRISEnv(
    K, N, q_bits,
    alpha0=1e-3, alphaK=5e-3,
    beta=0.5, noise_var=1.0
)

# =====================================================
# MARL models
# =====================================================

actors_marl = [Actor(4, N, q_bits) for _ in range(K)]
critic_marl = Critic(state_dim=4*K)

opt_actor_marl = torch.optim.Adam(
    [p for a in actors_marl for p in a.parameters()], lr=1e-3
)
opt_critic_marl = torch.optim.Adam(critic_marl.parameters(), lr=1e-3)

# =====================================================
# SARL models
# =====================================================

actor_sarl = Actor(4*K, N*K, q_bits)
critic_sarl = Critic(state_dim=4*K)

opt_actor_sarl = torch.optim.Adam(actor_sarl.parameters(), lr=1e-3)
opt_critic_sarl = torch.optim.Adam(critic_sarl.parameters(), lr=1e-3)

# =====================================================
# Training + logging
# =====================================================

returns_marl = []
returns_sarl = []

hs_power_marl = []
hs_power_sarl = []

for ep in range(EPISODES):

    # ---------- MARL ----------
    G_marl = train_marl(
        env_marl,
        actors_marl,
        critic_marl,
        opt_actor_marl,
        opt_critic_marl,
        T=5
    )
    returns_marl.append(G_marl)

    with torch.no_grad():
        obs = env_marl.reset()
        actions = []
        for k, a in enumerate(actors_marl):
            idx, _ = a.sample_action(a(obs[k]))
            actions.append(idx * (2 * torch.pi / (2 ** q_bits)))
        _, _, _, info = env_marl.step(actions)
        hs_power_marl.append(info["snr"])

    # ---------- SARL ----------
    G_sarl = train_sarl(
        env_sarl,
        actor_sarl,
        critic_sarl,
        opt_actor_sarl,
        opt_critic_sarl,
        T=5
    )
    returns_sarl.append(G_sarl)

    with torch.no_grad():
        obs = env_sarl.reset()
        obs_s = torch.cat(obs)
        idx, _ = actor_sarl.sample_action(actor_sarl(obs_s))
        phi_all = idx * (2 * torch.pi / (2 ** q_bits))
        actions = torch.split(phi_all, N)
        _, _, _, info = env_sarl.step(actions)
        hs_power_sarl.append(info["snr"])

    if ep % 100 == 0:
        print(
            f"Episode {ep}: "
            f"MARL G={G_marl:.3e}, "
            f"SARL G={G_sarl:.3e}"
        )

# =====================================================
# Moving average
# =====================================================

def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, mode="valid")

ma_marl = moving_average(np.array(returns_marl), MA_WINDOW)
ma_sarl = moving_average(np.array(returns_sarl), MA_WINDOW)

# =====================================================
# Plots
# =====================================================

plt.figure(figsize=(6, 4))
plt.plot(ma_marl, label="MARL (Return MA)")
plt.plot(ma_sarl, label="SARL (Return MA)")
plt.yscale("log")
plt.xlabel("Episode")
plt.ylabel("Moving Avg Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(
    moving_average(np.array(hs_power_marl), MA_WINDOW),
    label="MARL  |h_s|^2"
)
plt.plot(
    moving_average(np.array(hs_power_sarl), MA_WINDOW),
    label="SARL  |h_s|^2"
)
plt.yscale("log")
plt.xlabel("Episode")
plt.ylabel("Mean |h_s|^2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()