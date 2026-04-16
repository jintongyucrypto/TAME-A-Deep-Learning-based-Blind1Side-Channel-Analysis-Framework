import os
import math
import random
import scipy.io as spio
import copy
from copy import deepcopy
import h5py
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

def HW(s):
    return bin(int(s)).count("1")

def calculate_HW(data):
    hw = [bin(x).count("1") for x in range(256)]
    return [hw[int(s)] for s in data]

def sbox_label_cpa(plt, correct_key, leakage):
    num_traces = plt.shape[0]
    labels_for_snr = np.zeros(num_traces)
    for i in range(num_traces):
        if leakage == 'HW':
            labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key])
        elif leakage == 'ID':
            labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key]
    return labels_for_snr

def m_label_cpa(plt, leakage):
    num_traces = plt.shape[0]
    labels_for_snr = np.zeros(num_traces)
    for i in range(num_traces):
        if leakage == 'HW':
            labels_for_snr[i] = HW(plt[i])
        elif leakage == 'ID':
            labels_for_snr[i] = plt[i]
    return labels_for_snr

def cpa_method(total_samplept, number_of_traces, label, traces):
    if total_samplept == 1:
        cpa = np.zeros(total_samplept)
        cpa[0] = abs(np.corrcoef(label[:number_of_traces], traces[:number_of_traces])[1, 0])
    else:
        cpa = np.zeros(total_samplept)
        print("Calculate CPA!!")
        for t in range(total_samplept):
            cpa[t] = abs(np.corrcoef(label[:number_of_traces], traces[:number_of_traces, t])[1, 0])
    return cpa

def gen_trace(n_features, k = 0x03, var_noise = 0.1):
    trace = np.zeros(n_features)
    for j in range(trace.shape[0]):
        trace[j] = random.randint(0, 255)
    p = random.randint(0,255)
    label = AES_Sbox[p ^ k]
    for i in range(5,11):
        trace[i] = label
    noise_traces = np.zeros(n_features)
    for i in range(n_features):
        noise_traces = random.gauss(0, var_noise)
    trace = trace + noise_traces
    return trace, label, p

def gen_trace_mask_order1(n_features, k = 0x03, var_noise = 0.1):
    trace= np.zeros(n_features)
    for j in range(trace.shape[0]):
        trace[j] = random.randint(0,255)
    p = random.randint(0,255)
    m = random.randint(0,255)
    label = AES_Sbox[p ^ k]
    trace[10] = label ^ m
    trace[5] = m
    noise_traces = np.zeros(n_features)
    for i in range(n_features):
        noise_traces = random.gauss(0, var_noise)
    trace = trace + noise_traces
    return trace, label, p

def gen_trace_mask_order2(n_features, k = 0x03, var_noise = 0.1):
    trace= np.zeros(n_features)
    for j in range(trace.shape[0]):
        trace[j] = random.randint(0,255)
    p = random.randint(0,255)
    m1 = random.randint(0,255)
    m2 = random.randint(0,255)
    label = AES_Sbox[p ^ k]
    trace[10] = label ^ m1 ^ m2
    trace[5] = m1
    trace[8] = m2
    noise_traces = np.zeros(n_features)
    for i in range(n_features):
        noise_traces = random.gauss(0, var_noise)
    trace = trace + noise_traces
    return trace, label, p

def gen_trace_mask_order3(n_features, k = 0x03, var_noise = 0.1):
    trace= np.zeros(n_features)
    for j in range(trace.shape[0]):
        trace[j] = random.randint(0,255)
    p = random.randint(0,255)
    m1 = random.randint(0,255)
    m2 = random.randint(0,255)
    m3 = random.randint(0,255)
    label = AES_Sbox[p ^ k]
    trace[10] = label ^ m1 ^ m2 ^ m3
    trace[5] = m1
    trace[8] = m2
    trace[12] = m3
    noise_traces = np.zeros(n_features)
    for i in range(n_features):
        noise_traces = random.gauss(0, var_noise)
    trace = trace + noise_traces
    return trace, label, p

def generate_traces(n_traces,n_features= 20, order = 2):
    traces = np.zeros((n_traces, n_features))
    labels = np.zeros((n_traces,))
    plaintexts = np.zeros((n_traces))
    for i in tqdm(range(n_traces)):
        trace=None
        label=None
        p = None
        if order == 0:
            print("gen_trace_mask_order0")
            trace, label, p = gen_trace(n_features,k = 0x03, var_noise = 0.01)
        elif order == 1:
            print("gen_trace_mask_order1")
            trace, label, p = gen_trace_mask_order1(n_features,k = 0x03, var_noise = 0.01)
        elif order == 2:
            print("gen_trace_mask_order2")
            trace, label, p  = gen_trace_mask_order2(n_features,k = 0x03, var_noise = 0.01)
        elif order == 3:
            print("gen_trace_mask_order3")
            trace, label, p  = gen_trace_mask_order3(n_features,k = 0x03, var_noise = 0.01)
        traces[i, :] = trace
        labels[i] = label
        plaintexts[i] = p
    traces = traces.astype(np.int64)
    labels = labels.astype(np.int64)
    plaintexts = plaintexts.astype(np.uint8)
    return traces, labels, plaintexts

def gen_trace_blind(n_features,leakage = "HW", k = 0x03, var_noise = 0.1):
    print("var_noise: ", var_noise)
    trace = np.zeros(n_features)
    p = random.randint(0,255)

    label = AES_Sbox[p ^ k]

    if leakage == "ID":
        if n_features <25:
            trace[0] = p
            trace[1] = label
        elif n_features > 25:
            for i in range(0,10):
                trace[i] = p
            for j in range(15, 25):
                trace[j] = label

    elif leakage == "HW":
        if n_features < 25:
            trace[0] = HW(p)
            trace[1] = HW(label)
        elif n_features > 25:
            for i in range(0,10):
                trace[i] = HW(p)
            for j in range(15, 25):
                trace[j] = HW(label)
    trace = trace + np.random.normal(0, var_noise, n_features)
    return trace, label, p

def generate_traces_blind(n_traces, n_features=2, leakage="HW", order = 0,var_noise = 0.01):
    traces = np.zeros((n_traces, n_features))
    labels = np.zeros((2, n_traces,))
    plaintexts = np.zeros(n_traces)
    key = 0x03
    for i in tqdm(range(n_traces)):
        trace=None
        label=None
        p = None
        if order == 0:
            print("gen_trace_mask_order0")
            trace, label, p = gen_trace_blind(n_features,leakage = leakage,k = key, var_noise = var_noise)
        traces[i, :] = trace
        labels[0, i] = HW(p)
        labels[1, i] = HW(label)
        plaintexts[i] = p
    labels = labels.astype(np.int64)
    plaintexts = plaintexts.astype(np.uint8)
    return traces, labels, plaintexts, key, var_noise

def load_chipwhisperer_blind(chipwhisper_folder, byte=0, sync=True, noise_std=0.0, shift=10):
    if sync:
        X = np.load(chipwhisper_folder + 'traces.npy')[:10000]
        print("Synchronized dataset is loaded!")
    else:
        desyn_path = chipwhisper_folder + f'Desync_traces_{shift}.npy'
        if os.path.exists(desyn_path):
            X = np.load(desyn_path)
            print(f"Desynchronized dataset loaded from cache: {desyn_path}")
        else:
            print(f"Generating desynchronized dataset (max shift ±{shift})")
            X_orig = np.load(chipwhisper_folder + 'traces.npy')[:10000]
            shifted = np.zeros_like(X_orig)
            for j in range(X_orig.shape[0]):
                s = np.random.randint(-shift, shift + 1)
                if s > 0:
                    shifted[j, s:] = X_orig[j, :-s]
                elif s < 0:
                    shifted[j, :s] = X_orig[j, -s:]
                else:
                    shifted[j] = X_orig[j]
            X = shifted
            np.save(desyn_path, X)
            print(f"Desynchronized dataset saved to cache: {desyn_path}")
        print("Desynchronized data is loaded!")

    if noise_std > 0.0:
        noise = np.random.normal(0, noise_std, X.shape)
        X += noise
        print(f"Gaussian noise is added with standard deviation {noise_std}")

    P = np.load(chipwhisper_folder + 'plain.npy')[:10000, byte]
    HW_P = np.asarray(calculate_HW(P))
    labels = np.load(chipwhisper_folder + 'labels.npy')[:10000]
    HW_Y = np.asarray(calculate_HW(labels))
    keys = np.load(chipwhisper_folder + 'key.npy')
    return X, HW_Y, HW_P, keys[0][byte]

def calc_cpa(h,traces):
    c = np.zeros(traces.shape[1])
    for i in range(0,traces.shape[1]):
        c[i] = abs(np.corrcoef(h, traces[:,i])[0,1])
    return c

def winres(trace, window=20, overlap=0.5):
    trace_winres = []
    step = int(window * overlap)
    max = len(trace)
    for i in range(0, max, step):
        trace_winres.append(np.mean(trace[i:i + window]))
    return np.array(trace_winres)

def load_kyber_blind(saving_dataset_root, dataset_name, traces_no, window, load_reduced, sync=True):
    overlap = 0.5
    file_name = f'reduced_dataset_{window}.npy'
    file_path = f"{saving_dataset_root}{file_name}"
    if load_reduced:
        file_name_desyn = f'reduced_dataset_reduce{window}_desyn.npy'
    else:
        file_name_desyn = f'reduced_dataset_raw_desyn.npy'
    file_path_desyn = f"{saving_dataset_root}{file_name_desyn}"
    print("Dataset file path:", file_path if sync else file_path_desyn)

    traces_per_file = 100
    file_no = int(traces_no / traces_per_file)
    set_size = copy.deepcopy(traces_per_file)

    if not sync:
        if os.path.exists(file_path_desyn):
            print("loading desynchronized (random-shifted) trace set from cache")
            data = np.load(file_path_desyn)
            trace_kyber = data.astype("uint8")
            print("loaded dataset size: ", trace_kyber.shape)
        else:
            n_feats = 50000
            trace_kyber = np.zeros((file_no * set_size, n_feats), dtype="uint8")
            print("loading original trace set for desynchronization")
            for ii in range(0, file_no):
                path = saving_dataset_root + dataset_name + "/tracesA%d.mat" % ((ii + 1) * set_size - 1)
                mat = spio.loadmat(path, squeeze_me=True)
                traces = mat["tracesA"]
                idx = np.arange((ii) * set_size, (ii + 1) * set_size)
                trace_kyber[idx, :] = np.array(traces, dtype="uint8")
            print("Applying random shift (max ±20 samples) to each trace")
            max_shift = 20
            shifted = np.zeros_like(trace_kyber)
            for j in range(trace_kyber.shape[0]):
                shift = np.random.randint(-max_shift, max_shift + 1)
                if shift > 0:
                    shifted[j, shift:] = trace_kyber[j, :-shift]
                elif shift < 0:
                    shifted[j, :shift] = trace_kyber[j, -shift:]
                else:
                    shifted[j] = trace_kyber[j]
            trace_kyber = shifted
            if load_reduced:
                ns = int(trace_kyber.shape[1] / window) * 2
                new_trace_kyper = np.zeros((trace_kyber.shape[0], ns))
                print("window average the desynchronized trace")
                for j in range(trace_kyber.shape[0]):
                    new_trace_kyper[j, :] = winres(trace_kyber[j, :], window=window, overlap=overlap)
                trace_kyber = new_trace_kyper
            print("Saving desynchronized traces for the next time:")
            np.save(file_path_desyn, trace_kyber)

    elif not load_reduced:
        n_feats = 50000
        trace_kyber = np.zeros((file_no * set_size, n_feats), dtype="uint8")
        print("loading original trace set")
        for ii in range(0, file_no):
            path = saving_dataset_root + dataset_name + "/tracesA%d.mat" % ((ii + 1) * set_size - 1)
            mat = spio.loadmat(path, squeeze_me=True)
            traces = mat["tracesA"]
            idx = np.arange((ii) * set_size, (ii + 1) * set_size)
            trace_kyber[idx, :] = np.array(traces, dtype="uint8")

    elif load_reduced and not os.path.exists(file_path):
        n_feats = 50000
        trace_kyber = np.zeros((file_no * set_size, n_feats), dtype="uint8")
        print("loading original trace set")
        for ii in range(0, file_no):
            path = saving_dataset_root + dataset_name + "/tracesA%d.mat" % ((ii + 1) * set_size - 1)
            mat = spio.loadmat(path, squeeze_me=True)
            traces = mat["tracesA"]
            idx = np.arange((ii) * set_size, (ii + 1) * set_size)
            trace_kyber[idx, :] = np.array(traces, dtype="uint8")

        ns = int(len(traces[0]) / window) * 2
        new_trace_kyper = np.zeros((trace_kyber.shape[0], ns))
        print("window average the trace")
        for j in range(trace_kyber.shape[0]):
            new_trace_kyper[j, :] = winres(trace_kyber[j, :], window=window, overlap=overlap)
        print("trace_kyber shape after window average: ", trace_kyber.shape, window, overlap)

        print("Saving the reduced traces for the next time:")
        np.save(saving_dataset_root + f"reduced_dataset_{window}.npy", new_trace_kyper)
        trace_kyber = new_trace_kyper

    elif os.path.exists(file_path) and load_reduced:
        print("loading reduced trace set")
        data = np.load(file_path)
        trace_kyber = data.astype("uint8")
        print("loaded dataset size: ", trace_kyber.shape)
        print("finished loading trace set")

    vals = np.zeros((file_no * set_size, 12), dtype="uint8")

    print("Loading the secret key and messages")
    file_path = f"{saving_dataset_root}hw_val_inp.npy"
    if os.path.exists(file_path):
        data = np.load(file_path)
        hw_val_inp = data.astype("uint8")
        file_path = f"{saving_dataset_root}hw_val_out.npy"
        data = np.load(file_path)
        hw_val_out = data.astype("uint8")
        file_path = f"{saving_dataset_root}key.npy"
        key = np.load(file_path)
    else:
        for ii in range(0, file_no):
            idx = np.arange((ii) * set_size, (ii + 1) * set_size)
            path = saving_dataset_root + dataset_name + "/noncesA%d.mat" % ((ii + 1) * set_size - 1)
            mat = spio.loadmat(path, squeeze_me=True)
            val = mat["noncesA"]
            val = val.astype(np.uint8)
            vals[idx, :] = np.array(val, dtype="uint8")

        hw_val_inp = np.zeros(traces_no, dtype="uint8")
        for j in range(traces_no):
            hw_val_inp[j] = bin(int(vals[j, 2])).count("1") + bin(int(vals[j, 3])).count("1")

        hw_val_out = np.zeros(traces_no, dtype="uint8")
        for j in range(traces_no):
            hw_val_out[j] = bin(int(vals[j, 4])).count("1") + bin(int(vals[j, 5])).count("1")
        key = vals[0][1] * 256 + val[0][0]

        np.save(saving_dataset_root + f"hw_val_out.npy", hw_val_out)
        np.save(saving_dataset_root + f"hw_val_inp.npy", hw_val_inp)
        np.save(saving_dataset_root + f"key", key)

    print("Finished loading all")
    print(trace_kyber.shape)
    return trace_kyber, hw_val_out, hw_val_inp, key

def load_ascon_blind(root_data, dataset_name, num_traces, target=0, sync=True):
    import trsfile

    path_samples     = f"{root_data}ascon_samples.npy"
    path_samples_desyn = f"{root_data}ascon_samples_desyn.npy"
    path_labels      = f"{root_data}ascon_labels.npy"

    if os.path.exists(path_labels):
        label_data   = np.load(path_labels, allow_pickle=True).item()
        Y            = label_data['Y']
        P1           = label_data['P1']
        P2           = label_data['P2']
        correct_key  = label_data['correct_key']
        labels_ready = True
    else:
        labels_ready = False

    if sync:
        if os.path.exists(path_samples):
            print("loading Ascon trace set from cache")
            samples = np.load(path_samples)
            print("loaded dataset size:", samples.shape)
        else:
            print("parsing Ascon .trs file")
            ns = 772
            nonces  = np.zeros((num_traces, 16), dtype=np.uint8)
            keys    = np.zeros((num_traces, 16), dtype=np.uint8)
            raw     = np.zeros((num_traces, ns), dtype=np.float64)
            with trsfile.open(f'{root_data}{dataset_name}', 'r') as traces:
                for i, trace in enumerate(traces):
                    nonces[i] = trace.parameters['nonce'].value
                    keys[i]   = trace.parameters['key'].value
                    raw[i]    = trace
            samples = raw[100000:, :]
            np.save(path_samples, samples)
            print("Ascon traces saved to cache:", path_samples)
            if not labels_ready:
                correct_key = keys[100001][target]
                key0   = keys[100000:, target]
                nonce0 = nonces[100000:, target]
                nonce1 = nonces[100000:, target + 8]
                Y  = np.asarray(labelize_ascon(key0, nonce0, nonce1, target))
                P1 = np.asarray(calculate_HW(nonce0))
                P2 = np.asarray(calculate_HW(nonce1))
                np.save(path_labels, {'Y': Y, 'P1': P1, 'P2': P2, 'correct_key': correct_key})
                labels_ready = True

    else:
        if os.path.exists(path_samples_desyn):
            print("loading desynchronized Ascon trace set from cache")
            samples = np.load(path_samples_desyn)
            print("loaded dataset size:", samples.shape)
        else:
            if os.path.exists(path_samples):
                print("loading Ascon sync cache as base for desynchronization")
                samples = np.load(path_samples)
            else:
                print("parsing Ascon .trs file for desynchronization")
                ns = 772
                nonces  = np.zeros((num_traces, 16), dtype=np.uint8)
                keys    = np.zeros((num_traces, 16), dtype=np.uint8)
                raw     = np.zeros((num_traces, ns), dtype=np.float64)
                with trsfile.open(f'{root_data}{dataset_name}', 'r') as traces:
                    for i, trace in enumerate(traces):
                        nonces[i] = trace.parameters['nonce'].value
                        keys[i]   = trace.parameters['key'].value
                        raw[i]    = trace
                samples = raw[100000:, :]
                if not labels_ready:
                    correct_key = keys[100001][target]
                    key0   = keys[100000:, target]
                    nonce0 = nonces[100000:, target]
                    nonce1 = nonces[100000:, target + 8]
                    Y  = np.asarray(labelize_ascon(key0, nonce0, nonce1, target))
                    P1 = np.asarray(calculate_HW(nonce0))
                    P2 = np.asarray(calculate_HW(nonce1))
                    np.save(path_labels, {'Y': Y, 'P1': P1, 'P2': P2, 'correct_key': correct_key})
                    labels_ready = True
            print("Applying random shift (max ±20 samples) to each Ascon trace")
            max_shift = 20
            shifted = np.zeros_like(samples)
            for j in range(samples.shape[0]):
                shift = np.random.randint(-max_shift, max_shift + 1)
                if shift > 0:
                    shifted[j, shift:] = samples[j, :-shift]
                elif shift < 0:
                    shifted[j, :shift] = samples[j, -shift:]
                else:
                    shifted[j] = samples[j]
            samples = shifted
            np.save(path_samples_desyn, samples)
            print("Desynchronized Ascon traces saved to cache:", path_samples_desyn)

    if not labels_ready:
        label_data  = np.load(path_labels, allow_pickle=True).item()
        Y           = label_data['Y']
        P1          = label_data['P1']
        P2          = label_data['P2']
        correct_key = label_data['correct_key']

    print("Good key is:", correct_key)
    print("Finished loading Ascon dataset, shape:", samples.shape)
    return samples, Y, P1, P2, correct_key

def labelize_ascon(key1, nonce0, nonce1, target):
    iv = [128, 64, 12, 6, 0, 0, 0, 0]
    intermediate_values = nonce0 ^ nonce1 ^ (iv[target] & key1) ^ (nonce1 & key1) ^ key1
    y4 = [bin(iv).count("1") for iv in intermediate_values]
    return y4

def NTGE_fn(GE):
    NTGE = float('inf')
    for i in range(GE.shape[0] - 1, -1, -1):
        if GE[i] > 0:
            break
        elif GE[i] == 0:
            NTGE = i
    return NTGE

def create_noise_transition_matrix(classes, noise_rate, symmetric_noise = True):
    noise_transition_matrix = np.ones((classes, classes))

    if symmetric_noise == True:
        print("(noise_rate/(classes - 1)): ", (noise_rate/(classes - 1)))
        noise_transition_matrix = noise_transition_matrix * (noise_rate/(classes - 1))
    else:
        random_values = np.random.sample((classes, classes))
        random_sum = np.sum(random_values, axis= 0)
        for i in range(classes):
            random_sum[i] = random_sum[i] - random_values[i,i]
            noise_transition_matrix[i, :] = noise_rate*random_values[i,:]/random_sum[i]

    for i in range(classes):
        noise_transition_matrix[i, i] = (1 - noise_rate)

    print("new noise_transition_matrix:", noise_transition_matrix)
    print(np.sum(noise_transition_matrix, axis=0))
    return noise_transition_matrix

def make_discrete_hw(non_disc_hw, num_bits):
    disc_hws = np.zeros(len(non_disc_hw))
    for i in range(len(non_disc_hw)):
        frac, intNum = math.modf(non_disc_hw[i])
        if intNum > num_bits:
            disc_hws[i] = num_bits
        elif intNum < 0:
            disc_hws[i] = 0
        else:
            if frac >= 0.5 and intNum < num_bits:
                disc_hws[i] = intNum + 1
            else:
                disc_hws[i] = intNum
    return disc_hws

def compute_noise_transition_matrix_knownVsClavierlabel(Y, empirical_hws):
    noise_transition_matrix = np.zeros((9*9,9*9))
    print("Y", Y.shape)
    print("empirical_hws", empirical_hws)
    print("empirical_hws", empirical_hws.shape)
    for i in range(empirical_hws.shape[1]):
        col_index = int(empirical_hws[0, i]* 9+ empirical_hws[1, i])
        row_index = int(Y[0, i]* 9+ Y[1, i])
        noise_transition_matrix[row_index, col_index] += 1
    for j in range(noise_transition_matrix.shape[0]):
        if np.sum(noise_transition_matrix[j,:]) != 0:
            noise_transition_matrix[j,:] = noise_transition_matrix[j,:]/np.sum(noise_transition_matrix[j,:])
    print("noise_transition_matrix:", noise_transition_matrix)
    return noise_transition_matrix

def two_dim_hw_to_joint(two_dim_hw):
    joint_hw = two_dim_hw[:, 0] * 9 + two_dim_hw[:, 1]
    return joint_hw

def joint_hw_to_two_dim(joint_hw, classes):
    p = joint_hw // classes
    y = joint_hw % classes
    return p, y

def joint_hw_to_three_dim(joint_hw):
    p1 = joint_hw // 81
    p2 = (joint_hw % 81) // 9
    y = joint_hw % 9
    return p1, p2, y

def detailed_accuracy(p, y, p_emp, y_emp, classes):
    detailed_acc_y = np.zeros(classes)
    detailed_acc_m = np.zeros(classes)
    for j in range(classes):
        actual_y = 0
        accurate_y = 0
        for i in range(len(y)):
            if y[i] == j:
                actual_y += 1
                if y_emp[i] == y[i]:
                    accurate_y += 1
        if actual_y == 0:
            actual_y += 1

        actual_m = 0
        accurate_m = 0
        for i in range(len(p)):
            if p[i] == j:
                actual_m += 1
                if p_emp[i] == p[i]:
                    accurate_m += 1
        if actual_m == 0:
            actual_m += 1
        detailed_acc_y[j] = accurate_y / actual_y * 100
        detailed_acc_m[j] = accurate_m / actual_m * 100

    return detailed_acc_m, detailed_acc_y

def detailed_accuracy_ascon(p1, p2, y, p1_emp, p2_emp, y_emp):
    detailed_acc_m1 = np.zeros(9)
    detailed_acc_m2 = np.zeros(9)
    detailed_acc_y = np.zeros(9)
    for j in range(9):
        actual_y = 0
        accurate_y = 0
        for i in range(len(y)):
            if y[i] == j:
                actual_y += 1
                if y_emp[i] == y[i]:
                    accurate_y += 1
        if actual_y == 0:
            actual_y += 1

        actual_m1 = 0
        accurate_m1 = 0
        for i in range(len(p1)):
            if p1[i] == j:
                actual_m1 += 1
                if p1_emp[i] == p1[i]:
                    accurate_m1 += 1
        if actual_m1 == 0:
            actual_m1 += 1

        actual_m2 = 0
        accurate_m2 = 0
        for i in range(len(p2)):
            if p2[i] == j:
                actual_m2 += 1
                if p2_emp[i] == p2[i]:
                    accurate_m2 += 1
        if actual_m2 == 0:
            actual_m2 += 1
        detailed_acc_y[j] = accurate_y / actual_y * 100
        detailed_acc_m1[j] = accurate_m1 / actual_m1 * 100
        detailed_acc_m2[j] = accurate_m2 / actual_m2 * 100

    return detailed_acc_m1, detailed_acc_m2, detailed_acc_y

def total_accuracy(actual, empirical):
    accuracy = 0
    num_examples = actual.shape[0]

    for i in range(num_examples):
        if empirical[i] == actual[i]:
            accuracy += 1
    return accuracy/num_examples * 100

def single_var_accuracy(p1, p2, y,solo_hw_p1, solo_hw_p2, solo_hw_y, var_index=3):
    if var_index == 3:
        accuracy_m1, accuracy_m2, accuracy_y = 0, 0, 0
        num_examples = solo_hw_p1.shape[0]

        for i in range(num_examples):
            if p1[i] == solo_hw_p1[i]:
                accuracy_m1 += 1
            if p2[i] == solo_hw_p2[i]:
                accuracy_m2 += 1
            if y[i] == solo_hw_y[i]:
                accuracy_y += 1

    return accuracy_m1/num_examples * 100, accuracy_m2/num_examples * 100, accuracy_y/num_examples * 100

def customized_accuracy(predicted_hw, actual, classes):
    total_acc = total_accuracy(actual, predicted_hw)

    p, y = joint_hw_to_two_dim(actual, int(np.sqrt(classes)))
    solo_hw_p, solo_hw_y = joint_hw_to_two_dim(predicted_hw, int(np.sqrt(classes)))
    class_acc_m, class_acc_y = detailed_accuracy(p, y, solo_hw_p, solo_hw_y, int(np.sqrt(classes)))
    return total_acc, class_acc_m, class_acc_y

def customized_accuracy_ascon(predicted_hw, actual):
    total_acc = total_accuracy(actual, predicted_hw)

    p1, p2, y = joint_hw_to_three_dim(actual)
    solo_hw_p1, solo_hw_p2, solo_hw_y = joint_hw_to_three_dim(predicted_hw)
    acc_m1, acc_m2, acc_y = single_var_accuracy(p1, p2, y,solo_hw_p1, solo_hw_p2, solo_hw_y, var_index=3)
    class_acc_m1, class_acc_m2, class_acc_y = \
        detailed_accuracy_ascon(p1, p2, y, solo_hw_p1, solo_hw_p2, solo_hw_y)
    return total_acc, class_acc_m1, class_acc_m2, class_acc_y, acc_m1, acc_m2, acc_y

def weighted_majority_vote(row, weights):
    reverse_probabilities = [1 / w if w != 0 else 0 for w in weights]

    weighted_counts = {}

    for observation in row:
        if observation in weighted_counts:
            weighted_counts[int(observation)] += reverse_probabilities[int(observation)]
        else:
            weighted_counts[int(observation)] = reverse_probabilities[int(observation)]

    majority_class = max(weighted_counts, key=weighted_counts.get)
    return majority_class

def Linge_Clustering_centre(traces, class_combi, mean):
    sorted_index = np.argsort(traces, axis=0)
    if mean:
        num_classes = len(traces)
        HW = np.zeros(num_classes)
        num_per_class = class_combi.copy()
        class_considering = 0
        for idx in range(num_classes):
            while num_per_class[class_considering] < 1:
                class_considering += 1
            if num_per_class[class_considering] >= 1:
                HW[sorted_index[idx]] = class_considering
                num_per_class[class_considering] = num_per_class[class_considering] - 1
        return HW
    else:
        HW = np.zeros((traces.shape[0], traces.shape[1]))
        for point in range(traces.shape[1]):
            num_per_class = class_combi.copy()
            class_considering = 0
            for idx in range(traces.shape[0]):
                while num_per_class[class_considering] < 1:
                    class_considering += 1
                if num_per_class[class_considering] >= 1:
                    HW[sorted_index[idx][point]][point] = class_considering
                    num_per_class[class_considering] = num_per_class[class_considering] - 1

        majority = np.apply_along_axis(weighted_majority_vote, axis=1, arr=HW, weights=class_combi)
        return majority

def Linge_Clustering(traces_m, traces_y, num_clusters, dataset):
    mean = False
    if dataset == "Kyber":
        num_per_class = [1, 2, 3, 4, 8, 18, 33, 48, 55, 48, 33, 18, 8, 4, 3, 2, 1]
    elif dataset == "Chipwhisperer":
        num_per_class = [1, 2, 8, 18, 23, 18, 8, 2, 1]
    else:
        print("Dataset name is wrong!!")
        exit(-1)

    majority_m = Linge_Clustering_centre(traces_m, num_per_class, mean)
    majority_y = Linge_Clustering_centre(traces_y, num_per_class, mean)

    Y_pred = majority_y + math.sqrt(num_clusters) * majority_m
    return Y_pred

def Linge_Clustering_Ascon(traces_m1, traces_m2, traces_y, mean):
    num_per_class = [3, 23, 80, 159, 199, 159, 80, 23, 3]
    majority_m1 = Linge_Clustering_centre(traces_m1, num_per_class, mean)
    majority_m2 = Linge_Clustering_centre(traces_m2, num_per_class, mean)
    majority_y = Linge_Clustering_centre(traces_y, num_per_class, mean)

    Y_pred = 81 * majority_m1 + 9 * majority_m2 + majority_y
    return Y_pred