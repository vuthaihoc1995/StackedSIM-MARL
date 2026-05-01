import torch
import math


def rayleigh_channel(shape, scale):
    real = torch.randn(shape)
    imag = torch.randn(shape)
    return scale * (real + 1j * imag) / math.sqrt(2)