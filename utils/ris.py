import torch
import math


def quantize_phase(phi_cont, q_bits):
    levels = 2 ** q_bits
    step = 2 * math.pi / levels
    return torch.round(phi_cont / step) * step