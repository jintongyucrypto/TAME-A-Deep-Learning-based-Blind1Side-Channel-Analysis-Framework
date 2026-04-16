import math
import numpy as np
from tqdm import tqdm

from src.utils import HW

import cupy as cp
from tqdm import tqdm
import time

def attack_blind_log_gpu(hw, histogram, std):
    hw = cp.asarray(hw)
    hw0 = cp.asarray(hw[:, 0])
    hw1 = cp.asarray(hw[:, 1])
    mle = cp.zeros((len(histogram), hw.shape[0]))

    for k in range(len(histogram)):
        dist = cp.asarray(histogram[k])
        m, y = cp.meshgrid(cp.arange(dist.shape[0]), cp.arange(dist.shape[1]), indexing='ij')
        pr1 = dist[m, y]
        pr2 = 1 / (std * cp.sqrt(2 * cp.pi)) * cp.exp(-0.5 * ((hw0[:, cp.newaxis, cp.newaxis] - m) / std) ** 2)
        pr3 = 1 / (std * cp.sqrt(2 * cp.pi)) * cp.exp(-0.5 * ((hw1[:, cp.newaxis, cp.newaxis] - y) / std) ** 2)
        tpc = pr1 * pr2 * pr3
        tpc = cp.where(tpc < 0.00000001, 0.00000001, tpc)
        temp2 = cp.sum(tpc, axis=(1, 2))
        mle[k, :] = cp.cumsum(cp.log10(temp2))
    return mle.get()

def attack_blind_log_gpu_ascon(hw, histogram, std):
    hw = cp.asarray(hw)
    hw0 = cp.asarray(hw[:, 0])
    hw1 = cp.asarray(hw[:, 1])
    hw2 = cp.asarray(hw[:, 2])
    mle = cp.zeros((len(histogram), hw.shape[0]))
    for k in range(len(histogram)):
        dist = cp.asarray(histogram[k])
        x = cp.arange(9)
        y = cp.arange(9)
        z = cp.arange(9)
        m1, m2, y = cp.meshgrid(x, y, z, indexing='ij')
        pr1 = dist[m1, m2, y]
        pr2 = 1 / (std * cp.sqrt(2 * cp.pi)) * cp.exp(-0.5 * ((hw0[:, cp.newaxis, cp.newaxis, cp.newaxis] - m1) / std) ** 2)
        pr3 = 1 / (std * cp.sqrt(2 * cp.pi)) * cp.exp(-0.5 * ((hw1[:, cp.newaxis, cp.newaxis, cp.newaxis] - m2) / std) ** 2)
        pr4 = 1 / (std * cp.sqrt(2 * cp.pi)) * cp.exp(-0.5 * ((hw2[:, cp.newaxis, cp.newaxis, cp.newaxis] - y) / std) ** 2)
        tpc = pr1 * pr2 * pr3 * pr4
        tpc = cp.where(tpc < 0.00000001, 0.00000001, tpc)
        temp2 = cp.sum(tpc, axis=(1, 2, 3))
        mle[k, :] = cp.cumsum(cp.log10(temp2))
    return mle.get()

