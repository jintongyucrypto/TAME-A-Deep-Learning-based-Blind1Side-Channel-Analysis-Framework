import math
from collections import Counter
import numpy as np

from src.utils import make_discrete_hw

def majority_vote(row):
    counts = Counter(row)
    majority_class, _ = counts.most_common(1)[0]
    return majority_class

def reset(num_bits, shape):
    num_per_class_m = np.zeros(num_bits+1)
    num_per_class_y = np.zeros(num_bits+1)
    class_considering_m = 0
    class_considering_y = 0
    for j in range(num_bits + 1):
        num_per_class_m[j] = math.ceil(shape / (2 ** num_bits) * math.comb(num_bits, j))
        num_per_class_y[j] = math.ceil(shape/ (2 ** num_bits) * math.comb(num_bits, j))
        sume = 0
        for j in range(num_bits + 1):
            sume += num_per_class_m[j]
        if sume > shape:
            sub = (sume - shape) / 2
            num_per_class_m[0] -= sub
            num_per_class_m[num_bits] -= sub
            num_per_class_y[0] -= sub
            num_per_class_y[num_bits] -= sub

    return num_per_class_m, num_per_class_y, class_considering_m, class_considering_y

def linge_label(traces, poi, num_bits):
    sorted_index = np.argsort(traces[:, poi], axis=0)

    num_per_class = np.zeros(num_bits + 1)
    for j in range(num_bits + 1):
        num_per_class[j] = math.ceil(traces.shape[0] / (2 ** num_bits) * math.comb(num_bits, j))
    HW = np.zeros(traces.shape[0])

    class_considering = 0

    for idx in range(traces.shape[0]):
        while num_per_class[class_considering] < 1:
            class_considering += 1
        if num_per_class[class_considering] >= 1:
            HW[sorted_index[idx]] = class_considering
            num_per_class[class_considering] = num_per_class[class_considering] - 1
    return HW

def LinearRegression_VarianceAnalysis(traces_m, traces_y, num_bits, variance_noise):
    '''
    output: Y_pred = [HW_m, HW_y] with shape (2, number_of_trace)
    '''
    expectation_trace_m = np.mean(traces_m)
    expectation_trace_y = np.mean(traces_y)
    variance_trace_m = np.var(traces_m)
    variance_trace_y = np.var(traces_y)
    Alpha_m = np.sqrt((variance_trace_m - variance_noise) / (num_bits / 4))
    Alpha_y = np.sqrt((variance_trace_y - variance_noise) / (num_bits / 4))
    Beta_m = expectation_trace_m - 4 * Alpha_m
    Beta_y = expectation_trace_y - 4 * Alpha_y
    HW_m = np.abs((traces_m - Beta_m) / Alpha_m)
    HW_y = np.abs((traces_y - Beta_y) / Alpha_y)
    HW_m = make_discrete_hw(HW_m, num_bits)
    HW_y = make_discrete_hw(HW_y, num_bits)

    return HW_m, HW_y

def LinearRegression_VarianceAnalysis_Ascon(traces_m1, traces_m2, traces_y, num_bits, var_noise=0.001):
    '''
    output: Y_pred = [HW_m, HW_y] with shape (2, number_of_trace)
    '''
    expectation_trace_m1 = np.mean(traces_m1)
    expectation_trace_m2 = np.mean(traces_m2)
    expectation_trace_y = np.mean(traces_y)
    variance_trace_m1 = np.var(traces_m1)
    variance_trace_m2 = np.var(traces_m2)
    variance_trace_y = np.var(traces_y)
    Alpha_m1 = np.sqrt((variance_trace_m1 - var_noise) / (num_bits / 4))
    Alpha_m2 = np.sqrt((variance_trace_m2 - var_noise) / (num_bits / 4))
    Alpha_y = np.sqrt((variance_trace_y - var_noise) / (num_bits / 4))
    Beta_m1 = expectation_trace_m1 - 4 * Alpha_m1
    Beta_m2 = expectation_trace_m2 - 4 * Alpha_m2
    Beta_y = expectation_trace_y - 4 * Alpha_y
    HW_m1 = np.abs((traces_m1 - Beta_m1) / Alpha_m1)
    HW_m2 = np.abs((traces_m2 - Beta_m2) / Alpha_m2)
    HW_y = np.abs((traces_y - Beta_y) / Alpha_y)
    HW_m1 = make_discrete_hw(HW_m1, num_bits)
    HW_m2 = make_discrete_hw(HW_m2, num_bits)
    HW_y = make_discrete_hw(HW_y, num_bits)
    return HW_m1, HW_m2, HW_y

def Linge_label_methodology(traces, poi_m, poi_y, num_bits=8):
    HW_m = linge_label(traces, poi_m, num_bits)
    HW_y = linge_label(traces, poi_y, num_bits)
    return HW_m, HW_y

def Linge_label_methodology_Ascon(traces, poi_m1, poi_m2, poi_y, num_bits=8):
    HW_m1 = linge_label(traces, poi_m1, num_bits)
    HW_m2 = linge_label(traces, poi_m2, num_bits)
    HW_y = linge_label(traces, poi_y, num_bits)
    return HW_m1, HW_m2, HW_y

