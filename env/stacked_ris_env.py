import torch
import math
from utils.channel import rayleigh_channel


class StackedRISEnv:
    """
    Stacked RIS environment with:
    - Channel reset every episode
    - Properly scaled reward
    - Noise variance fixed (can be 1)
    """

    def __init__(self, K, N, q_bits,
                 alpha0, alphaK, beta,
                 noise_var=1.0):
        self.K = K
        self.N = N
        self.q_bits = q_bits

        # Large-scale fading parameters
        self.alpha0 = alpha0
        self.alphaK = alphaK
        self.beta = beta

        # Noise variance (can be fixed to 1)
        self.noise_var = noise_var

        # Initialize channel + RIS
        self._reset_channel()
        self.phi = [torch.zeros(self.N) for _ in range(self.K)]

    # --------------------------------------------------
    # Channel reset (CRITICAL FIX)
    # --------------------------------------------------
    def _reset_channel(self):
        """Generate a new channel realization"""
        self.h0 = rayleigh_channel((self.N, 1), self.alpha0)
        self.hK = rayleigh_channel((self.N, 1), self.alphaK)
        self.H = [
            rayleigh_channel((self.N, self.N), self.beta)
            for _ in range(self.K - 1)
        ]

    # --------------------------------------------------
    # RL interface
    # --------------------------------------------------
    def reset(self):
        """
        Reset environment:
        - New channel realization
        - Zero RIS phases
        """
        self._reset_channel()
        self.phi = [torch.zeros(self.N) for _ in range(self.K)]
        return self._get_observations()

    def step(self, actions):
        """
        actions: list of K tensors, each of shape [N]
        """
        # Apply RIS phases
        for k in range(self.K):
            self.phi[k] = actions[k]

        # Effective channel
        hs = self.compute_effective_channel()

        # Instantaneous SNR (noise_var can be 1)
        snr = (torch.abs(hs) ** 2).real / self.noise_var

        # --------------------------------------------------
        # REWARD FIX (MOST IMPORTANT)
        # Use log-scaled reward for stable learning
        # --------------------------------------------------
        reward = 1e8 * snr.item()

        obs = self._get_observations(hs)

        done = False
        info = {
            "snr": snr.item(),          # |h_s|^2 / noise_var
            "hs_power": snr.item()
        }

        return obs, reward, done, info

    # --------------------------------------------------
    # Physics: effective stacked RIS channel
    # --------------------------------------------------
    def compute_effective_channel(self):
        """
        h_s = h_K^H Phi_K H_{K-1} ... H_1 Phi_1 h_0
        """
        x = self.h0

        for k in range(self.K):
            phi_complex = torch.exp(
                1j * self.phi[k].to(torch.cfloat)
            )
            Phi = torch.diag(phi_complex)

            if k < self.K - 1:
                x = self.H[k] @ Phi @ x
            else:
                x = Phi @ x

        return self.hK.conj().T @ x  # (1x1) complex scalar

    # --------------------------------------------------
    # Observation
    # --------------------------------------------------
    def _get_observations(self, hs=None):
        if hs is None:
            hs = self.compute_effective_channel()

        obs = []
        for _ in range(self.K):
            obs_k = torch.tensor([
                torch.norm(self.h0).item(),
                torch.norm(self.hK).item(),
                hs.real.item(),
                hs.imag.item()
            ])
            obs.append(obs_k)

        return obs
