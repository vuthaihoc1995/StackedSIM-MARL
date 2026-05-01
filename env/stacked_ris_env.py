import torch
from utils.channel import rayleigh_channel


class StackedRISEnv:
    def __init__(self, K, N, q_bits, alpha0, alphaK, beta, noise_var):
        self.K = K
        self.N = N
        self.q_bits = q_bits
        self.noise_var = noise_var

        # Channels
        self.h0 = rayleigh_channel((N, 1), alpha0)
        self.hK = rayleigh_channel((N, 1), alphaK)
        self.H = [rayleigh_channel((N, N), beta) for _ in range(K - 1)]

        # RIS phases
        self.phi = [torch.zeros(N) for _ in range(K)]

    def reset(self):
        self.phi = [torch.zeros(self.N) for _ in range(self.K)]
        return self._get_observations()

    def step(self, actions):
        for k in range(self.K):
            self.phi[k] = actions[k]

        hs = self.compute_effective_channel()

        # Real scalar SNR
        snr = (torch.abs(hs) ** 2).real / self.noise_var
        reward = snr.item()

        obs = self._get_observations(hs)

        done = False
        info = {"snr": reward}
        return obs, reward, done, info

    def compute_effective_channel(self):
        x = self.h0
        for k in range(self.K):
            phi_c = torch.exp(1j * self.phi[k].to(torch.cfloat))
            Phi = torch.diag(phi_c)
            if k < self.K - 1:
                x = self.H[k] @ Phi @ x
            else:
                x = Phi @ x
        return self.hK.conj().T @ x  # (1×1) complex

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
