import torch
import numpy as np
import matplotlib.pyplot as plt

from env.stacked_ris_env import StackedRISEnv
from marl.agent import Actor


# =====================================================
# Evaluation functions
# =====================================================

def evaluate_op_sarl(env, actor, snr_db, gamma_th, num_mc=10000):
    snr_linear = 10 ** (snr_db / 10)
    outages = 0

    actor.eval()

    with torch.no_grad():
        for _ in range(num_mc):
            obs = env.reset()
            obs_sarl = torch.cat(obs)

            logits = actor(obs_sarl)
            idx, _ = actor.sample_action(logits)

            phi_all = idx * (2 * torch.pi / (2 ** actor.q_bits))
            actions = torch.split(phi_all, env.N)

            _, _, _, info = env.step(actions)
            gamma_m = snr_linear * info["snr"]

            if gamma_m < gamma_th:
                outages += 1

    return outages / num_mc


def evaluate_op_marl(env, actors, snr_db, gamma_th, num_mc=10000):
    snr_linear = 10 ** (snr_db / 10)
    outages = 0

    for a in actors:
        a.eval()

    with torch.no_grad():
        for _ in range(num_mc):
            obs = env.reset()
            actions = []

            for k, actor in enumerate(actors):
                logits = actor(obs[k])
                idx, _ = actor.sample_action(logits)
                phi = idx * (2 * torch.pi / (2 ** actor.q_bits))
                actions.append(phi)

            _, _, _, info = env.step(actions)
            gamma_m = snr_linear * info["snr"]

            if gamma_m < gamma_th:
                outages += 1

    return outages / num_mc


# =====================================================
# System setup (MUST match training)
# =====================================================

K, N, q_bits = 2, 8, 2

env = StackedRISEnv(
    K=K, N=N, q_bits=q_bits,
    alpha0=1e-3, alphaK=5e-3,
    beta=0.5, noise_var=1e-3
)

# ---- Load trained policies (replace with your checkpoints if saved) ----
actor_sarl = Actor(obs_dim=4*K, N=N*K, q_bits=q_bits)
actors_marl = [Actor(obs_dim=4, N=N, q_bits=q_bits) for _ in range(K)]

# =====================================================
# OP vs SNR evaluation
# =====================================================

snr_db_range = np.arange(0, 31, 5)
R = 0.25
gamma_th = 2**R - 1
num_mc = 10000

op_sarl = []
op_marl = []

for snr_db in snr_db_range:
    print(f"Evaluating at SNR = {snr_db} dB")

    op_sarl.append(
        evaluate_op_sarl(env, actor_sarl, snr_db, gamma_th, num_mc)
    )

    op_marl.append(
        evaluate_op_marl(env, actors_marl, snr_db, gamma_th, num_mc)
    )

# =====================================================
# Plot
# =====================================================

plt.figure()
plt.semilogy(snr_db_range, op_sarl, 'o-', label='SARL')
plt.semilogy(snr_db_range, op_marl, 's-', label='MARL')
plt.xlabel('Average SNR (dB)')
plt.ylabel('Outage Probability')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.show()