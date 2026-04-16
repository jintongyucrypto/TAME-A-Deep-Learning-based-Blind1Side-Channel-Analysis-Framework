import os.path
import numpy as np

from src.utils import AES_Sbox, HW

def AES_Hypothetical_Distribution_Model(saving_path):
    histogram_path = os.path.join(saving_path + "histogram_AES.npy")
    if os.path.exists(histogram_path):
        H = np.load(histogram_path)
    else:
        H = np.zeros((256, 9, 9))
        print(H.shape)
        total_sample = 0
        for k in range(256):
            for m in range(256):
                H[k,HW(m), HW(AES_Sbox[m^k])] += 1
                total_sample += 1
        H = H/total_sample
        np.savez(saving_path + "histogram_AES", H)
    return H

Q_REF = 0xD01
QINV_REF = -3327

def hamming_weight(n):
    binary_string = bin(n)
    c = binary_string.count('1')
    return c

def sign_extend(value, num_bits):
    sign_bit = 1 << (num_bits - 1)
    mask = (1 << num_bits) - 1
    sign_extended = value & mask
    if value & sign_bit:
        sign_extended |= ~mask
    return sign_extended

def montgomery_reduce(a):
    t = sign_extend((sign_extend(a, 16) * QINV_REF), 16)
    t = (sign_extend(a, 32) - sign_extend(t * Q_REF, 32)) >> 16
    return t

def Kyber_Hypothetical_Distribution_Model(saving_path):
    histogram_path = os.path.join(saving_path+"histogram_kyber.npy")
    if os.path.exists(histogram_path):
        histogram = np.load(histogram_path)
    else:
        data_range = np.arange(-(Q_REF - 1) / 2, (Q_REF - 1) / 2 + 1)
        data_range2 = np.arange(Q_REF)

        HW_list = []
        n_bits = 16

        for a in data_range2:
            HW_array = np.zeros((n_bits + 1, n_bits + 1))
            for b in data_range:
                p = montgomery_reduce(sign_extend(int(b), 16) * sign_extend(int(a), 16))
                b_hw = hamming_weight(int(b) & 0xFFFF)
                p_hw = hamming_weight(int(p) & 0xFFFF)
                HW_array[b_hw, p_hw] += 1

            HW_list.append(HW_array / np.sum(HW_array))
        histogram = np.asarray(HW_list)
        np.save(saving_path+"histogram_kyber.npy", histogram)
    return histogram

def Ascon_Hypothetical_Distribution_Model(saving_path, target):
    histogram_path = os.path.join(saving_path + "histogram_Ascon.npy")
    if os.path.exists(histogram_path):
        H = np.load(histogram_path)
    else:
        H = np.zeros((256, 9, 9, 9))
        iv = [128, 64, 12, 6, 0, 0, 0, 0]
        print(H.shape)
        total_sample = 0
        for k in range(256):
            for m1 in range(256):
                for m2 in range(256):
                    H[k, HW(m1), HW(m2), HW(m1 ^ m2 ^ (iv[target] & k) ^ (m2 & k) ^ k)] += 1
                total_sample += 1
        H = H/total_sample
        np.save(saving_path + "histogram_Ascon.npy", H)
    return H

