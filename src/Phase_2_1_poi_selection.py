import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from src.utils import sbox_label_cpa, cpa_method, m_label_cpa

def cpa_m_and_y_poi_selection_kyber(X, P, Y, plot_cpa, image_root, dataset, poi):
    total_sample = X.shape[1]
    number_of_traces = X.shape[0]
    cpa_y = cpa_method(total_sample, number_of_traces, Y, traces=X)
    cpa_y = np.nan_to_num(cpa_y)
    poi_y = np.argsort(cpa_y)[-poi:][::-1]
    cpa_m = cpa_method(total_sample, number_of_traces, P, traces=X)
    cpa_m = np.nan_to_num(cpa_m)
    poi_m = np.argsort(cpa_m)[-poi:][::-1]

    if plot_cpa == True:
        print("We are ready to plot POI")
        fig, ax = plt.subplots(figsize=(15, 10))
        x_axis = [i for i in range(total_sample)]
        ax.plot(x_axis, cpa_y, c="red", label=r"Coefficients multiplication, $a_0.b_0$")
        ax.plot(x_axis, cpa_m, c="blue", label="Message, $m$")
        ax.set_xlabel('Sample points', fontsize=20)
        ax.set_ylabel('(Absolute) Correlation', fontsize=20)
        for label_fig in (ax.get_xticklabels() + ax.get_yticklabels()):
            label_fig.set_fontsize(20)
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=3, prop={'size': 20})
        plt.savefig(image_root + 'CA_' + dataset + "_m_and_y.png")
    return poi_m, poi_y

def cpa_m_and_y_poi_selection_Ascon(X, P1, P2, Y, plot_cpa, image_root, poi):
    total_samplept = X.shape[1]
    number_of_traces = X.shape[0]

    cpa_y = cpa_method(total_samplept, number_of_traces, Y, traces=X)
    cpa_y = np.nan_to_num(cpa_y)
    poi_y = np.argsort(cpa_y)[-poi:][::-1]

    cpa_m1 = cpa_method(total_samplept, number_of_traces, P1, traces=X)
    cpa_m1 = np.nan_to_num(cpa_m1)
    poi_m1 = np.argsort(cpa_m1)[-poi:][::-1]

    cpa_m2 = cpa_method(total_samplept, number_of_traces, P2, traces=X)
    cpa_m2 = np.nan_to_num(cpa_m2)
    poi_m2 = np.argsort(cpa_m2)[-poi:][::-1]

    if plot_cpa == True:
        fig, ax = plt.subplots(figsize=(15, 10))
        x_axis = [i for i in range(total_samplept)]
        ax.plot(x_axis, cpa_y, c="red", label="y")
        ax.plot(x_axis, cpa_m1, c="blue", label="$m_0$")
        ax.plot(x_axis, cpa_m2, c="green", label="$m_1$")
        ax.set_xlabel('Sample points', fontsize=30)
        ax.set_ylabel('(Absolute) Correlation', fontsize=30)
        for label_fig in (ax.get_xticklabels() + ax.get_yticklabels()):
            label_fig.set_fontsize(30)
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=3, prop={'size': 30})
        plt.savefig(image_root + 'CPA_n0_n1_y.png')
    return poi_m1, poi_m2, poi_y

def cpa_m_and_y_poi_selection(X, P, Y, plot_cpa, image_root, poi):
    total_samplept = X.shape[1]
    number_of_traces = X.shape[0]

    cpa_y = cpa_method(total_samplept, number_of_traces, Y, traces=X)
    cpa_y = np.nan_to_num(cpa_y)
    poi_y = np.argsort(cpa_y)[-poi:][::-1]

    cpa_m = cpa_method(total_samplept, number_of_traces, P, traces=X)
    cpa_m = np.nan_to_num(cpa_m)
    poi_m = np.argsort(cpa_m)[-poi:][::-1]

    print("poi_m: ", poi_m)
    print("cpa_y[poi_y]", cpa_y[poi_y])
    print("cpa_m[poi_m]", cpa_m[poi_m])
    if plot_cpa == True:
        fig, ax = plt.subplots(figsize=(15, 10))
        x_axis = [i for i in range(total_samplept)]
        ax.plot(x_axis, cpa_y, c="red", label="$Sbox(m \oplus k^*)$")
        ax.plot(x_axis, cpa_m, c="blue", label="Plaintext, $m$")
        ax.set_xlabel('Sample points', fontsize=20)
        ax.set_ylabel('(Absolute) Correlation', fontsize=20)
        for label_fig in (ax.get_xticklabels() + ax.get_yticklabels()):
            label_fig.set_fontsize(20)
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=3, prop={'size': 20})
        plt.savefig(image_root +'CPA_m_and_y.png')
    return poi_m, poi_y

def variance_m_and_y_poi_selection(X, poi):
    variance = np.var(X, axis=0)
    indices = np.arange(len(variance))

    top_indices_m = indices[np.argsort(variance)[-poi:][::-1]]
    top_indices_y = indices[np.argsort(variance)[-poi:][::-1]]

    return top_indices_m, top_indices_y

