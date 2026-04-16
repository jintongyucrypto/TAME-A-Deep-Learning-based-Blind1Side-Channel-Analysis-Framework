# DL-BSCA: Deep Learning Blind Side-Channel Analysis with DTCR and TAME

This repository contains the open-source implementation accompanying our paper. It provides two frameworks for **blind side-channel analysis (BSCA)** — attacking cryptographic implementations without knowledge of the secret key during the profiling phase — applied to four target ciphers: **AES** (synchronized and desynchronized traces), **Kyber**, and **Ascon**.

---

## Overview

Blind SCA assumes the attacker has access only to unlabeled power traces during training. The proposed pipeline comprises three phases:

1. **Phase 1 — Theoretical Distribution**: Pre-compute the theoretical joint Hamming-weight distribution over all candidate key hypotheses.
2. **Phase 2 — Empirical Distribution via Labeling**: Select Points of Interest (PoIs) using CPA, then assign pseudo-labels to unlabeled traces via one of several labeling strategies (VA, slicing, GMM clustering, or DTCR-based clustering).
3. **Phase 3 — Maximum Likelihood Key Recovery**: Compare the empirical distribution predicted by the DNN against the theoretical distribution for each key candidate using a log-likelihood ratio, accumulating scores over multiple traces to rank keys.

The key contribution of this work is twofold:
- **DTCR-based multi-point clustering** (labeling strategy `Mixture_DTCR`): replaces GMM with a Deep Temporal Clustering Representation encoder trained jointly with a K-means objective and a real/fake discriminator, yielding more accurate pseudo-labels for clustered traces.
- **TAME (Trace-Adaptive Model Estimation)**: integrates Learning to Re-weight Examples (LRE) into the DNN training phase. A small set of clean, confidently-labeled traces (selected from cluster centers) is used as a validation signal to automatically learn per-sample weights, suppressing the influence of mislabeled training traces.

---

## Frameworks and Target Ciphers

### Framework 1: DTCR + DNN (CNN / MLP)

Standard blind SCA pipeline where DTCR provides improved pseudo-labels and a DNN (MLP or CNN) is trained on those labels without any noise-correction mechanism.

| Script | Target | Notes |
|---|---|---|
| `DL-BSCA_DTCR+CNN_kyber_aes.py` | AES (synchronized), Kyber | `dataset` variable selects `'Chipwhisperer'` or `'Kyber'` |
| `DL-BSCA_DTCR+CNN_Ascon.py` | Ascon | Three intermediate variables: HW(m1), HW(m2), HW(y) |

### Framework 2: DTCR + TAME

Extends Framework 1 with LRE-based sample re-weighting during DNN training. A small clean validation set is constructed from cluster centers (GMM or DTCR) and used to guide weight learning, mitigating label noise.

| Script | Target | Notes |
|---|---|---|
| `DL-BSCA_DTCR+TAME_kyber_aes.py` | AES (synchronized), Kyber | `dataset` selects `'Chipwhisperer'` or `'Kyber'` |
| `DL-BSCA_DTCR+TAME_aes_desyn.py` | AES (desynchronized) | `sync=False`; supports configurable random time shift via `--shift` |
| `DL-BSCA_DTCR+TAME_Ascon.py` | Ascon | Three-variable joint attack with LRE |

---

## Repository Structure

```
open_code_tame/
├── DL-BSCA_DTCR+CNN_Ascon.py          # DTCR+DNN attack on Ascon
├── DL-BSCA_DTCR+CNN_kyber_aes.py      # DTCR+DNN attack on AES / Kyber
├── DL-BSCA_DTCR+TAME_Ascon.py         # DTCR+TAME attack on Ascon
├── DL-BSCA_DTCR+TAME_aes_desyn.py     # DTCR+TAME attack on desynchronized AES
├── DL-BSCA_DTCR+TAME_kyber_aes.py     # DTCR+TAME attack on AES / Kyber
└── src/
    ├── Phase_1_create_hypothetical_model_distribution.py  # Theoretical HW distributions
    ├── Phase_2_1_poi_selection.py                         # CPA/variance PoI selection
    ├── Phase_2_VA_Slicing_labeling.py                     # VA and slicing labeling
    ├── Phase_3_maximum_likelihood.py                      # Log-likelihood key ranking
    ├── neural_networks.py                                 # MLP and CNN training (standard)
    ├── lre_mlp.py                                         # MLP and CNN training with LRE (TAME)
    └── utils.py                                           # Data loaders, accuracy metrics, NTGE
```

---

## Command-Line Arguments

All main scripts share the following arguments:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_idx` | int | 0 | Index of the model in the random-model pool |
| `--model_type` | str | `mlp` | Neural network type: `mlp` or `cnn` |
| `--labeling_type` | str | `Mixture_DTCR` | Labeling strategy: `ClavierLabel`, `LingeLabel`, `Mixture`, `Mixture_DTCR` |
| `--dropout` | str | `False` | Enable dropout: `True` or `False` |
| `--gpu` | int | 0 | CUDA device ID |

TAME scripts additionally accept:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--nb_val` | int | 200 | Number of clean traces for the LRE validation set |
| `--val_seed` | int | 42 | Random seed for validation set selection |
| `--val_source` | str | `gmm_center` | Validation selection strategy: `random`, `gmm_center`, `dtcr_center` |

The desynchronized AES script additionally accepts:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--shift` | int | 20 | Maximum random time shift applied to traces |

---

## Labeling Strategies

| Name | Method | Reference |
|---|---|---|
| `ClavierLabel` | Variance Analysis (VA) with linear regression | Clavier et al. |
| `LingeLabel` | Slicing / single-PoI threshold labeling | Linge et al. |
| `Mixture` | Gaussian Mixture Model (GMM) clustering over joint PoIs | This work (baseline) |
| `Mixture_DTCR` | Deep Temporal Clustering Representation (DTCR) + K-means | **This work (proposed)** |

DTCR jointly optimizes three objectives: (1) denoising sequence reconstruction, (2) K-means spectral relaxation, and (3) real/fake sample discrimination — learning compact, cluster-friendly representations of the PoI segments.

---

## Dependencies

```
tensorflow >= 2.x   (with tensorflow.compat.v1 for DTCR graph construction)
numpy
scikit-learn
```

---

## Evaluation Metrics

- **Guessing Entropy (GE)**: average rank of the correct key over repeated attack experiments.
- **Number of Traces for Guessing Entropy = 1 (NTGE)**: minimum number of traces needed for GE to reach 1.
- **Success Rate (SR)**: fraction of experiments where the correct key is ranked within the top-*k* candidates.

## Reference

Our work is built upon the paper *Breaking the Blindfold: Deep Learning-based Blind Side-channel Analysis* [1]. We sincerely appreciate the authors for providing their reproduction code, which we reuse and further improve upon in this work.

[1] Azade Rezaeezade, Trevor Yap, Dirmanto Jap, Shivam Bhasin, and Stjepan Picek. Breaking the blindfold: Deep learning-based blind side-channel analysis. In 34th USENIX Security Symposium (USENIX Security 25), pages 5777–5796, 2025.