import math
import os.path
import random
import argparse
import logging
import tensorflow as tf
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
np.set_printoptions(suppress=True)

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.Phase_1_create_hypothetical_model_distribution import AES_Hypothetical_Distribution_Model, \
    Kyber_Hypothetical_Distribution_Model
from src.Phase_2_1_poi_selection import cpa_m_and_y_poi_selection, variance_m_and_y_poi_selection
from src.Phase_2_VA_Slicing_labeling import LinearRegression_VarianceAnalysis, Linge_label_methodology
from src.Phase_3_maximum_likelihood import attack_blind_log_gpu
from src.utils import load_chipwhisperer_blind, NTGE_fn, customized_accuracy, joint_hw_to_two_dim, \
    load_kyber_blind, Linge_Clustering
from src.neural_networks import run_cnn, run_mlp
from src.lre_mlp import run_mlp_lre, run_cnn_lre

parser = argparse.ArgumentParser(description='Run DL_Joint, short version')
parser.add_argument('--model_idx', type=str, default=0, help='model ID')
parser.add_argument('--dropout', type=str, default=0, help='dropout')
parser.add_argument('--model_type', type=str, default=0, help='model_type')
parser.add_argument('--labeling_type', type=str, default=0, help='labeling_type')
parser.add_argument('--gpu', type=int, default=1, help='GPU id to use')
parser.add_argument('--nb_val', type=int, default=200,
                    help='number of clean labelled traces used as LRE validation set')
parser.add_argument('--val_seed', type=int, default=42,
                    help='random seed for selecting the clean validation indices')
parser.add_argument('--val_source', type=str, default='gmm_center',
                    choices=['random', 'gmm_center', 'dtcr_center'],
                    help='validation set selection strategy: '
                         '"random" = uniform random sampling from training set; '
                         '"gmm_center" = samples nearest to GMM cluster centres '
                         '(only effective when labeling_type=Mixture); '
                         '"dtcr_center" = samples nearest to K-means cluster centres '
                         'in DTCR representation space '
                         '(only effective when labeling_type=Mixture_DTCR)')
parser.add_argument('--shift', type=int, default=20, help='shift value')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
model_idx = int(args.model_idx)
shift = args.shift
dropout = args.dropout
if dropout == "True":
    dropout = True
else:
    dropout = False
model_type = args.model_type
labeling_type = args.labeling_type
nb_val   = args.nb_val
val_seed = args.val_seed
val_source = args.val_source

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

dataset = 'Chipwhisperer'
desyn = 'desyn'
loss_type = "CCE"
poi_selection_mode = 'cpa'
plot_cpa = False

nb_attacks = 100
epochs = 100
num_poi = 50
noise_std = 0.05

root_data = f".../{dataset}/"
root_results = f".../Results/{dataset}_{desyn}{shift}/"

print("dataset_root:", root_data)
os.makedirs(root_results, exist_ok=True)

if dataset == 'Chipwhisperer':
    byte = 0
    nb_traces_training = 8000
    nb_traces_attack = 2000
    num_bits = 8
    n_clusters = (num_bits + 1) * (num_bits + 1)
    sync = False
    X, Y, P, correct_key = load_chipwhisperer_blind(root_data, byte, sync, noise_std, shift)
    X_train = X[:nb_traces_training, :]
    Y_train = Y[:nb_traces_training]
    P_train = P[:nb_traces_training]
    X_test = X[nb_traces_training:, :]
    Y_test = Y[nb_traces_training:]
    P_test = P[nb_traces_training:]

elif dataset == "Kyber":
    dataset_name = "Reference-PPM"
    total_traces = 100000
    nb_traces_attack = 20000
    nb_traces_training = 80000
    num_bits = 16
    n_clusters = (num_bits + 1) * (num_bits + 1)
    window = 10
    use_reduced = True
    X, Y, P, correct_key = load_kyber_blind(root_data, dataset_name, total_traces, window, use_reduced)
    X_train = X[:nb_traces_training, :]
    Y_train = Y[:nb_traces_training]
    P_train = P[:nb_traces_training]
    X_test = X[nb_traces_training:, :]
    Y_test = Y[nb_traces_training:]
    P_test = P[nb_traces_training:]

print("Computing the theoretical joint distribution for all the possible keys!")
if dataset == "Kyber":
    theoretical_histogram = Kyber_Hypothetical_Distribution_Model(root_results)
else:
    theoretical_histogram = AES_Hypothetical_Distribution_Model(root_results)

if os.path.exists(root_results + f"{dataset}_{poi_selection_mode}_{num_poi}.npz"):
    data = np.load(root_results + f"{dataset}_{poi_selection_mode}_{num_poi}.npz", allow_pickle=True)
    poi_m = data["poi_m"]
    poi_y = data["poi_y"]

else:
    if poi_selection_mode == 'cpa':
        poi_m, poi_y = cpa_m_and_y_poi_selection(X_train, P_train, Y_train, plot_cpa, root_results, num_poi)
        np.savez(root_results + f"{dataset}_{poi_selection_mode}_{num_poi}",
                 poi_m=poi_m,
                 poi_y=poi_y,
                 )
    elif poi_selection_mode == "variance":
        poi_m, poi_y = variance_m_and_y_poi_selection(X_train, num_poi)
        np.savez(root_results + f"{dataset}_{poi_selection_mode}_{num_poi}",
                 poi_m=poi_m,
                 poi_y=poi_y,
                 )
    else:
        print("The poi_selection_mode is not supported!")
        exit(-1)
print("-------------------")
print("Dataset: ", dataset, "poi_selection_mode: ", poi_selection_mode, "Number of PoI: ", len(poi_m))

var_noise = 0.001
var_t = np.zeros(X_train.shape[1])
for i in range(len(var_t)):
    var_t[i] = np.var(X_train[:, i], dtype=np.float64)
    if var_t[i] != 0 and var_noise > var_t[i]:
        var_noise = var_t[i]
print("Var noise: ", var_noise)

joint_hw_train = P_train * (num_bits + 1) + Y_train
joint_hw_test = P_test * (num_bits + 1) + Y_test

print("labeling_type is:", labeling_type)
_dtcr_clusters_path = root_results + f"Mixture_DTCR_clusters_{dataset}_{poi_selection_mode}_{num_poi}.npy"
_dtcr_dists_path    = root_results + f"Mixture_DTCR_dists_{dataset}_{poi_selection_mode}_{num_poi}.npy"
_label_cached   = os.path.exists(root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy")
_cluster_cached = (labeling_type != 'Mixture_DTCR' or val_source != 'dtcr_center'
                   or (os.path.exists(_dtcr_clusters_path) and os.path.exists(_dtcr_dists_path)))

if _label_cached and _cluster_cached:
    combined_hws = np.load(
        root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy")
    solo_hw_p, solo_hw_y = joint_hw_to_two_dim(combined_hws, num_bits + 1)

else:
    if labeling_type == 'LingeLabel':
        solo_hw_p, solo_hw_y = Linge_label_methodology(X_train, poi_m[0], poi_y[0], num_bits)
        combined_hws = solo_hw_p * (num_bits + 1) + solo_hw_y
        np.save((root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy"),
                combined_hws)
    elif labeling_type == 'ClavierLabel':
        solo_hw_p, solo_hw_y = LinearRegression_VarianceAnalysis(X_train[:, poi_m[0]],
                                                                 X_train[:, poi_y[0]],
                                                                 num_bits, variance_noise=var_noise)

        combined_hws = solo_hw_p * (num_bits + 1) + solo_hw_y
        np.save((root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy"),
                combined_hws)
    elif labeling_type == 'Mixture':
        print("n_clusters: ", n_clusters)
        X_train_joint = np.hstack(([X_train[:, poi_m], X_train[:, poi_y]]))

        model_joint = GaussianMixture(n_components=n_clusters)
        model_joint.fit(X_train_joint)

        clusters_joint = model_joint.predict(X_train_joint)

        np.save(root_results + f"Mixture_clusters_{dataset}_{poi_selection_mode}_{num_poi}.npy",
                clusters_joint)
        np.save(root_results + f"Mixture_cluster_means_{dataset}_{poi_selection_mode}_{num_poi}.npy",
                model_joint.means_)

        avg_y = np.zeros((n_clusters, num_poi))
        avg_m = np.zeros((n_clusters, num_poi))
        for cluster in range(n_clusters):
            cluster_members = X_train_joint[clusters_joint == cluster]
            avg_m[cluster] = np.mean(cluster_members[:, 0:num_poi], axis=0)
            avg_y[cluster] = np.mean(cluster_members[:, num_poi:2 * num_poi], axis=0)

        empirical_hws = Linge_Clustering(avg_y, avg_m, n_clusters, dataset)

        cluster_hw = np.zeros(X_train_joint.shape[0], dtype="uint8")
        for cluster in range(n_clusters):
            new_label = empirical_hws[cluster]
            for i in range(X_train.shape[0]):
                if clusters_joint[i] == cluster:
                    cluster_hw[i] = new_label
        np.save((root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy"),
                cluster_hw)
        combined_hws = cluster_hw
        solo_hw_p, solo_hw_y = joint_hw_to_two_dim(combined_hws, num_bits + 1)

    elif labeling_type == 'Mixture_DTCR':
        print("n_clusters: ", n_clusters)
        X_train_joint = np.hstack(([X_train[:, poi_m], X_train[:, poi_y]]))

        import tensorflow.compat.v1 as tf1
        tf1.disable_eager_execution()

        def _dtcr_rnn_reformat(x, input_dims, n_steps):
            """Convert (batch, n_steps, input_dims) → list of n_steps tensors
            each of shape (batch, input_dims)."""
            x_ = tf1.transpose(x, [1, 0, 2])
            x_ = tf1.reshape(x_, [-1, input_dims])
            return tf1.split(x_, n_steps, 0)

        def _dtcr_construct_cells(hidden_structs, cell_type='GRU'):
            cells = []
            for h in hidden_structs:
                cells.append(tf1.nn.rnn_cell.GRUCell(h))
            return cells

        def _dtcr_dRNN(cell, inputs, rate, scope='default'):
            """Single-layer dilated RNN."""
            n_steps = len(inputs)
            if rate < 0 or rate >= n_steps:
                raise ValueError("dRNN rate is out of range.")
            EVEN = (n_steps % rate) == 0
            if not EVEN:
                zero_tensor = tf1.zeros_like(inputs[0])
                dialated_n_steps = n_steps // rate + 1
                for _ in range(dialated_n_steps * rate - n_steps):
                    inputs.append(zero_tensor)
            else:
                dialated_n_steps = n_steps // rate

            dilated_inputs = [
                tf1.concat(inputs[i * rate:(i + 1) * rate], axis=0)
                for i in range(dialated_n_steps)
            ]
            dilated_outputs, _ = tf1.nn.static_rnn(
                cell, dilated_inputs, dtype=tf1.float32, scope=scope)
            splitted = [tf1.split(o, rate, axis=0) for o in dilated_outputs]
            unrolled = [o for sub in splitted for o in sub]
            return unrolled[:n_steps]

        def _dtcr_multi_dRNN(cells, inputs, dilations):
            import copy
            output_list = []
            x = copy.copy(inputs)
            for idx, (cell, dilation) in enumerate(zip(cells, dilations)):
                scope_name = "multi_dRNN_dilation_%d" % idx
                x = _dtcr_dRNN(cell, x, dilation, scope=scope_name)
                x_trans = tf1.stack(x, axis=0)
                x_trans = tf1.transpose(x_trans, [1, 0, 2])
                output_list.append(x_trans)
            return x, output_list

        def _dtcr_drnn_layer_final(x, hidden_structs, dilations,
                                   n_steps, input_dims, cell_type='GRU'):
            assert len(hidden_structs) == len(dilations)
            x_reformat = _dtcr_rnn_reformat(x, input_dims, n_steps)
            cells = _dtcr_construct_cells(hidden_structs, cell_type)
            outputs, output = _dtcr_multi_dRNN(cells, x_reformat, dilations)
            return outputs, output

        def _dtcr_get_fake_sample(data):
            """Replace 20% of time steps per sample with values from other
            positions; return (fake_data, one-hot real/fake labels [2N,2])."""
            N, L = data.shape
            mask = np.ones((N, L), dtype=np.float32)
            rand_list = np.zeros((N, L), dtype=np.float32)
            n_fake = max(1, int(L * 0.2))
            fake_pos = np.random.randint(0, L, size=(N, n_fake))
            for i in range(N):
                for j in range(n_fake):
                    mask[i, fake_pos[i, j]] = 0
            for i in range(N):
                count = 0
                for j in range(L):
                    if j in fake_pos[i]:
                        rand_list[i, j] = data[i, fake_pos[i, count]]
                        count += 1
            fake_data = data * mask + rand_list * (1 - mask)
            real_fake_labels = np.zeros((N * 2, 2), dtype=np.float32)
            real_fake_labels[:N, 0] = 1.0
            real_fake_labels[N:, 1] = 1.0
            return fake_data, real_fake_labels

        dtcr_hidden_size = [100, 50, 50]
        dtcr_dilations   = [1, 2, 4]
        dtcr_cell_type   = 'GRU'
        dtcr_lamda       = 1.0
        dtcr_lr          = 1e-4
        dtcr_epochs      = 300
        dtcr_l2_coef     = 1e-4

        N_dtcr   = X_train_joint.shape[0]
        K_dtcr   = n_clusters
        L_dtcr   = X_train_joint.shape[1]
        emb_size = 1

        dtcr_scaler = StandardScaler()
        X_dtcr = dtcr_scaler.fit_transform(X_train_joint).astype(np.float32)

        print(f"DTCR: N={N_dtcr}, L={L_dtcr}, K={K_dtcr}, epochs={dtcr_epochs}")

        dtcr_batch_size = 256

        tf1.reset_default_graph()
        gpu_config = tf1.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        with tf1.Graph().as_default() as dtcr_graph:

            inp_ph       = tf1.placeholder(tf1.float32, [None, L_dtcr], name='inputs')
            noise_ph     = tf1.placeholder(tf1.float32, [None, L_dtcr], name='noise')
            rfl_ph       = tf1.placeholder(tf1.float32, [None, 2],      name='real_fake_label')
            batch_idx_ph = tf1.placeholder(tf1.int32,   [None],         name='batch_idx')
            F_new_ph     = tf1.placeholder(tf1.float32, [None, K_dtcr], name='F_new_value')

            F_var = tf1.get_variable(
                'F', shape=[N_dtcr, K_dtcr],
                initializer=tf1.orthogonal_initializer(),
                trainable=False)
            F_update_op = tf1.scatter_update(F_var, batch_idx_ph, F_new_ph)

            inputs_3d = tf1.reshape(inp_ph,   [-1, L_dtcr, emb_size])
            noises_3d = tf1.reshape(noise_ph, [-1, L_dtcr, emb_size])

            noise_input  = inputs_3d + noises_3d
            rev_noise    = tf1.reverse(noise_input, axis=[1])

            decoder_inputs = _dtcr_rnn_reformat(noise_input, emb_size, L_dtcr)
            targets        = _dtcr_rnn_reformat(inputs_3d,   emb_size, L_dtcr)

            total_hidden = int(np.sum(dtcr_hidden_size))

            with tf1.variable_scope('fw'):
                _, enc_out_fw = _dtcr_drnn_layer_final(
                    noise_input, dtcr_hidden_size, dtcr_dilations,
                    L_dtcr, emb_size, dtcr_cell_type)

            with tf1.variable_scope('bw'):
                _, enc_out_bw = _dtcr_drnn_layer_final(
                    rev_noise, dtcr_hidden_size, dtcr_dilations,
                    L_dtcr, emb_size, dtcr_cell_type)

            fw_parts = [enc_out_fw[i][:, -1, :] for i in range(len(dtcr_hidden_size))]
            bw_parts = [enc_out_bw[i][:, -1, :] for i in range(len(dtcr_hidden_size))]
            enc_fw = tf1.concat(fw_parts, axis=1)
            enc_bw = tf1.concat(bw_parts, axis=1)
            hidden_abstract = tf1.concat([enc_fw, enc_bw], axis=1)

            dec_cell_inner = tf1.nn.rnn_cell.GRUCell(total_hidden * 2)
            with tf1.variable_scope('decoder_proj'):
                w_out = tf1.get_variable(
                    'proj_w_out', [total_hidden * 2, emb_size],
                    initializer=tf1.random_uniform_initializer(-0.04, 0.04))
                b_out = tf1.get_variable(
                    'proj_b_out', [emb_size],
                    initializer=tf1.random_uniform_initializer(-0.04, 0.04))

            dec_outputs = []
            dec_state   = hidden_abstract
            prev_inp    = decoder_inputs[0]
            with tf1.variable_scope('decoder_rnn', reuse=tf1.AUTO_REUSE):
                for t in range(L_dtcr):
                    dec_out, dec_state = dec_cell_inner(prev_inp, dec_state)
                    prev_inp = tf1.matmul(dec_out, w_out) + b_out
                    dec_outputs.append(prev_inp)
            dec_outputs_tensor = tf1.stack(dec_outputs, axis=1)

            real_hidden = tf1.split(hidden_abstract, 2)[0]
            real_targets_tensor = tf1.split(
                tf1.stack(targets, axis=1), 2)[0]
            real_dec_tensor     = tf1.split(dec_outputs_tensor, 2)[0]

            loss_reconstruct = tf1.losses.mean_squared_error(
                labels=real_targets_tensor, predictions=real_dec_tensor)

            F_batch = tf1.gather(F_var, batch_idx_ph)
            W_km    = tf1.transpose(real_hidden)
            WTW     = tf1.matmul(real_hidden, W_km)
            FTWTWF  = tf1.matmul(tf1.matmul(tf1.transpose(F_batch), WTW), F_batch)
            loss_kmeans = tf1.linalg.trace(WTW) - tf1.linalg.trace(FTWTWF)

            d_hidden = hidden_abstract.get_shape().as_list()[1]
            W1 = tf1.get_variable('disc_w1', [d_hidden, 128],
                                  initializer=tf1.truncated_normal_initializer(stddev=0.01))
            b1 = tf1.get_variable('disc_b1', [128],
                                  initializer=tf1.constant_initializer(0.1))
            W2 = tf1.get_variable('disc_w2', [128, 2],
                                  initializer=tf1.truncated_normal_initializer(stddev=0.01))
            b2 = tf1.get_variable('disc_b2', [2],
                                  initializer=tf1.constant_initializer(0.1))
            disc_hidden = tf1.nn.relu(tf1.matmul(hidden_abstract, W1) + b1)
            disc_logits = tf1.matmul(disc_hidden, W2) + b2
            loss_disc = tf1.reduce_mean(
                tf1.nn.softmax_cross_entropy_with_logits(
                    logits=disc_logits, labels=rfl_ph))

            reg_loss = tf1.add_n(
                [tf1.nn.l2_loss(v) for v in tf1.trainable_variables()])

            loss_total = (loss_reconstruct
                          + dtcr_lamda / 2.0 * loss_kmeans
                          + loss_disc
                          + dtcr_l2_coef * reg_loss)

            train_op = tf1.train.AdamOptimizer(dtcr_lr).minimize(loss_total)

            init_op = tf1.global_variables_initializer()

        with tf1.Session(graph=dtcr_graph, config=gpu_config) as sess:
            sess.run(init_op)

            noise_std_dtcr = 0.1
            for epoch in range(dtcr_epochs):
                perm = np.random.permutation(N_dtcr)
                epoch_loss = 0.0
                n_batches  = 0
                for start in range(0, N_dtcr, dtcr_batch_size):
                    idx_b  = perm[start : start + dtcr_batch_size]
                    x_real = X_dtcr[idx_b]
                    b      = len(idx_b)
                    noise_b = np.random.normal(
                        0, noise_std_dtcr, size=(b * 2, L_dtcr)).astype(np.float32)
                    fake_b, rfl_b = _dtcr_get_fake_sample(x_real)
                    feed = {
                        inp_ph:       np.concatenate([x_real, fake_b], axis=0),
                        noise_ph:     noise_b,
                        rfl_ph:       rfl_b,
                        batch_idx_ph: idx_b,
                    }
                    loss_val, _ = sess.run([loss_total, train_op], feed_dict=feed)
                    epoch_loss += loss_val
                    n_batches  += 1

                if epoch % 50 == 0:
                    print(f"  DTCR epoch {epoch:3d}/{dtcr_epochs}  "
                          f"avg_loss={epoch_loss/n_batches:.4f}")

                if epoch % 10 == 0 and epoch != 0:
                    h_all = np.zeros((N_dtcr, total_hidden * 2), dtype=np.float32)
                    for start in range(0, N_dtcr, dtcr_batch_size):
                        idx_b  = np.arange(start, min(start + dtcr_batch_size, N_dtcr))
                        x_real = X_dtcr[idx_b]
                        b      = len(idx_b)
                        fake_b, rfl_b = _dtcr_get_fake_sample(x_real)
                        feed_h = {
                            inp_ph:       np.concatenate([x_real, fake_b], axis=0),
                            noise_ph:     np.zeros((b * 2, L_dtcr), dtype=np.float32),
                            rfl_ph:       rfl_b,
                            batch_idx_ph: idx_b,
                        }
                        h_all[idx_b] = sess.run(real_hidden, feed_dict=feed_h)
                    W_np  = h_all.T
                    _, _, VT = np.linalg.svd(W_np, full_matrices=False)
                    topk = VT[:K_dtcr, :].T
                    if topk.shape[0] < N_dtcr:
                        topk = np.vstack([
                            topk,
                            np.zeros((N_dtcr - topk.shape[0], K_dtcr), dtype=np.float32)
                        ])
                    for start in range(0, N_dtcr, dtcr_batch_size):
                        idx_b = np.arange(start, min(start + dtcr_batch_size, N_dtcr))
                        sess.run(F_update_op, feed_dict={
                            batch_idx_ph: idx_b,
                            F_new_ph:     topk[idx_b].astype(np.float32),
                        })

            representations = np.zeros((N_dtcr, total_hidden * 2), dtype=np.float32)
            for start in range(0, N_dtcr, dtcr_batch_size):
                idx_b  = np.arange(start, min(start + dtcr_batch_size, N_dtcr))
                x_real = X_dtcr[idx_b]
                b      = len(idx_b)
                fake_b, rfl_b = _dtcr_get_fake_sample(x_real)
                feed_f = {
                    inp_ph:       np.concatenate([x_real, fake_b], axis=0),
                    noise_ph:     np.zeros((b * 2, L_dtcr), dtype=np.float32),
                    rfl_ph:       rfl_b,
                    batch_idx_ph: idx_b,
                }
                representations[idx_b] = sess.run(real_hidden, feed_dict=feed_f)

        print("DTCR training complete. Running K-means on learned representations...")

        kmeans = KMeans(n_clusters=K_dtcr, random_state=42, n_init=10)
        clusters_joint = kmeans.fit_predict(representations)

        _dtcr_clusters_path = root_results + f"Mixture_DTCR_clusters_{dataset}_{poi_selection_mode}_{num_poi}.npy"
        _dtcr_dists_path    = root_results + f"Mixture_DTCR_dists_{dataset}_{poi_selection_mode}_{num_poi}.npy"
        _dtcr_per_sample_dists = np.linalg.norm(
            representations - kmeans.cluster_centers_[clusters_joint], axis=1)
        np.save(_dtcr_clusters_path, clusters_joint)
        np.save(_dtcr_dists_path,    _dtcr_per_sample_dists)

        avg_y = np.zeros((n_clusters, num_poi))
        avg_m = np.zeros((n_clusters, num_poi))
        for cluster in range(n_clusters):
            cluster_members = X_train_joint[clusters_joint == cluster]
            if len(cluster_members) == 0:
                continue
            avg_m[cluster] = np.mean(cluster_members[:, 0:num_poi], axis=0)
            avg_y[cluster] = np.mean(cluster_members[:, num_poi:2 * num_poi], axis=0)

        empirical_hws = Linge_Clustering(avg_y, avg_m, n_clusters, dataset)

        cluster_hw = np.zeros(X_train_joint.shape[0], dtype="uint8")
        for cluster in range(n_clusters):
            new_label = empirical_hws[cluster]
            cluster_hw[clusters_joint == cluster] = new_label

        np.save((root_results + f"{labeling_type}_empirical_hw_{dataset}_{poi_selection_mode}_{num_poi}.npy"),
                cluster_hw)
        combined_hws = cluster_hw
        solo_hw_p, solo_hw_y = joint_hw_to_two_dim(combined_hws, num_bits + 1)
    else:
        print("labeling_type is not supported!!")
        exit(-1)

print(f"Accuracy using {labeling_type} technique!")
labeling_total_acc, labeling_acc_m, labeling_acc_y = \
    customized_accuracy(combined_hws, joint_hw_train, num_bits + 1)

if val_source == 'gmm_center' and labeling_type == 'Mixture':
    _cluster_path = root_results + f"Mixture_clusters_{dataset}_{poi_selection_mode}_{num_poi}.npy"
    _means_path   = root_results + f"Mixture_cluster_means_{dataset}_{poi_selection_mode}_{num_poi}.npy"

    if os.path.exists(_cluster_path) and os.path.exists(_means_path):
        _clusters_joint    = np.load(_cluster_path)
        _cluster_means_gmm = np.load(_means_path)
        print("[val_source=gmm_center] 已从缓存加载聚类分配与 GMM 均值。")
    else:
        print("[val_source=gmm_center] 聚类缓存不存在，重新拟合 GMM ...")
        _X_tmp = np.hstack((X_train[:, poi_m], X_train[:, poi_y]))
        _gmm_tmp = GaussianMixture(n_components=n_clusters)
        _gmm_tmp.fit(_X_tmp)
        _clusters_joint    = _gmm_tmp.predict(_X_tmp)
        _cluster_means_gmm = _gmm_tmp.means_
        np.save(_cluster_path, _clusters_joint)
        np.save(_means_path,   _cluster_means_gmm)

    _X_joint      = np.hstack((X_train[:, poi_m], X_train[:, poi_y]))
    k_per_cluster = max(1, -(-nb_val // n_clusters))
    _candidates   = []
    for _c in range(n_clusters):
        _idx_c = np.where(_clusters_joint == _c)[0]
        if len(_idx_c) == 0:
            continue
        _dists   = np.linalg.norm(_X_joint[_idx_c] - _cluster_means_gmm[_c], axis=1)
        _k       = min(k_per_cluster, len(_idx_c))
        _nearest = _idx_c[np.argsort(_dists)[:_k]]
        _candidates.extend(_nearest.tolist())

    _candidates = np.array(_candidates)
    if len(_candidates) >= nb_val:
        _rng_val    = np.random.default_rng(val_seed)
        val_indices = _rng_val.choice(_candidates, size=nb_val, replace=False)
    else:
        val_indices = _candidates
        print(f"[WARNING] 候选样本数 {len(_candidates)} < 请求 {nb_val}，"
              f"将使用全部 {len(_candidates)} 个候选样本。")

    print(f"[ME] GMM 中心验证集：{len(val_indices)} 条轨迹 "
          f"（请求 {nb_val}，共 {n_clusters} 个聚类，每聚类最多 {k_per_cluster} 条）")

elif val_source == 'dtcr_center' and labeling_type == 'Mixture_DTCR':
    _dtcr_clusters_path = root_results + f"Mixture_DTCR_clusters_{dataset}_{poi_selection_mode}_{num_poi}.npy"
    _dtcr_dists_path    = root_results + f"Mixture_DTCR_dists_{dataset}_{poi_selection_mode}_{num_poi}.npy"

    if os.path.exists(_dtcr_clusters_path) and os.path.exists(_dtcr_dists_path):
        _dtcr_clusters = np.load(_dtcr_clusters_path)
        _dtcr_dists    = np.load(_dtcr_dists_path)
        print("[val_source=dtcr_center] 已从缓存加载 DTCR 聚类分配与距离。")
    else:
        raise RuntimeError(
            "DTCR 聚类缓存文件不存在。请先以 labeling_type='Mixture_DTCR' 运行一次以生成缓存，"
            f"期望路径：\n  {_dtcr_clusters_path}\n  {_dtcr_dists_path}")

    k_per_cluster = max(1, -(-nb_val // n_clusters))
    _candidates   = []
    for _c in range(n_clusters):
        _idx_c   = np.where(_dtcr_clusters == _c)[0]
        if len(_idx_c) == 0:
            continue
        _dists_c = _dtcr_dists[_idx_c]
        _k       = min(k_per_cluster, len(_idx_c))
        _nearest = _idx_c[np.argsort(_dists_c)[:_k]]
        _candidates.extend(_nearest.tolist())

    _candidates = np.array(_candidates)
    if len(_candidates) >= nb_val:
        _rng_val    = np.random.default_rng(val_seed)
        val_indices = _rng_val.choice(_candidates, size=nb_val, replace=False)
    else:
        val_indices = _candidates
        print(f"[WARNING] 候选样本数 {len(_candidates)} < 请求 {nb_val}，"
              f"将使用全部 {len(_candidates)} 个候选样本。")

    print(f"[ME] DTCR 中心验证集：{len(val_indices)} 条轨迹 "
          f"（请求 {nb_val}，共 {n_clusters} 个聚类，每聚类最多 {k_per_cluster} 条）")
else:
    if val_source == 'gmm_center':
        print(f"[WARNING] val_source='gmm_center' 仅支持 labeling_type='Mixture'，"
              f"当前为 '{labeling_type}'")
    _rng_val    = np.random.default_rng(val_seed)
    val_indices = _rng_val.choice(nb_traces_training, size=nb_val, replace=False)
    print(f"[ME] 随机验证集：{nb_val} 条轨迹 "
          f"（seed={val_seed}，索引范围 [{val_indices.min()}, {val_indices.max()}]）")

val_assigned_labels = combined_hws[val_indices]
val_true_labels     = joint_hw_train[val_indices]
val_label_acc, val_label_acc_m, val_label_acc_y = \
    customized_accuracy(val_assigned_labels, val_true_labels, num_bits + 1)

print(f"[Val 集标签质量]  总体: {val_label_acc:.4f}  "
      f"HW(m): {val_label_acc_m.mean():.4f}  HW(y): {val_label_acc_y.mean():.4f}")
print(f"[对比整体训练集]  总体: {labeling_total_acc:.4f}  "
      f"HW(m): {labeling_acc_m.mean():.4f}  HW(y): {labeling_acc_y.mean():.4f}")

_val_stats_path = (root_results +
                   f"val_stats_{labeling_type}_{val_source}_{nb_val}_{val_seed}.npy")
np.save(_val_stats_path, {
    "val_indices":         val_indices,
    "nb_val_actual":       len(val_indices),
    "nb_val_requested":    nb_val,
    "val_source":          val_source,
    "val_label_acc":       val_label_acc,
    "val_label_acc_m":     val_label_acc_m,
    "val_label_acc_y":     val_label_acc_y,
    "overall_label_acc":   labeling_total_acc,
    "overall_label_acc_m": labeling_acc_m,
    "overall_label_acc_y": labeling_acc_y,
})
print(f"[Val 集] 统计信息已保存至 {_val_stats_path}")

scaler_std = StandardScaler()
X_train = scaler_std.fit_transform(X_train)
X_test = scaler_std.transform(X_test)

X_val       = X_train[val_indices]
y_val_clean = combined_hws[val_indices]
print(f"[ME] 验证集已构建：{len(val_indices)} 条轨迹（来源: {val_source}）")

trained_model_root = root_results + f'{model_type}_{labeling_type}_{num_poi}_{epochs}_TAMEv1/'
base_models = root_results + f"{model_type}_base_models"

os.makedirs(trained_model_root + "trained_models", exist_ok=True)

for loss_type in ["CCE"]:
    generate_or_load = "load"

    _pred_cache_path = (trained_model_root +
                        f"predictions_{labeling_type}_{loss_type}_{nb_val}_{dropout}_{model_idx}.npy")

    if os.path.exists(_pred_cache_path):
        print(f"[Cache] 加载已缓存的预测结果，跳过 DNN 推理: {_pred_cache_path}")
        _cache = np.load(_pred_cache_path, allow_pickle=True).item()
        jointed_predicted_hw = _cache["jointed_predicted_hw"]
        accuracy = _cache["accuracy"]
    else:
        print("DNN model type:", model_type)
        if model_type == "mlp":
            jointed_predicted_hw, accuracy = run_mlp_lre(
                X_train, X_test,
                combined_hws,
                X_val,
                y_val_clean,
                joint_hw_train,
                joint_hw_test,
                epochs, n_clusters, trained_model_root, base_models, model_idx,
                dropout, loss_type, generate_or_load, dataset)

        elif model_type == "cnn":
            jointed_predicted_hw, accuracy = run_cnn_lre(
                X_train, X_test,
                combined_hws,
                X_val,
                y_val_clean,
                joint_hw_train,
                joint_hw_test,
                epochs, n_clusters, trained_model_root, base_models, model_idx,
                dropout, loss_type, generate_or_load, dataset)

        else:
            print("Neural Network type is not supported")
            exit(-1)

        jointed_predicted_hw = np.asarray(jointed_predicted_hw)
        np.save(_pred_cache_path, {"jointed_predicted_hw": jointed_predicted_hw, "accuracy": accuracy})
        print(f"[Cache] 预测结果已保存: {_pred_cache_path}")

    jointed_predicted_hw = np.asarray(jointed_predicted_hw)

    preds0 = jointed_predicted_hw // (num_bits + 1)
    preds1 = jointed_predicted_hw % (num_bits + 1)
    predicted_hw = np.array([preds0, preds1])

    if dataset == "Kyber":
        sr_order = 10
    else:
        sr_order = 1
    ge_rounds = np.zeros((nb_attacks, nb_traces_attack))
    ge = np.zeros(nb_traces_attack)
    success_rate_sum = np.zeros(nb_traces_attack)
    for i in range(nb_attacks):
        idx_trace = np.arange(predicted_hw.shape[1])
        np.random.shuffle(idx_trace)
        mle = attack_blind_log_gpu(predicted_hw[:, idx_trace[:nb_traces_attack]].T, theoretical_histogram,
                                   std=math.sqrt(var_noise))
        final_rank = np.zeros(nb_traces_attack)
        for j in range(nb_traces_attack):
            tmp_idx = np.argsort(mle[:, j])[::-1]
            final_rank[j] = np.float32(np.where(tmp_idx == correct_key)[0][0])
            if final_rank[j] <= sr_order:
                success_rate_sum[j] += 1
        ge_rounds[i] = final_rank
        ge += final_rank

    ge = ge / nb_attacks
    success_rate = success_rate_sum / nb_attacks
    print("GE: ", ge)

    NTGE = NTGE_fn(ge)
    print("NTGE:", NTGE)
    np.save(trained_model_root + f"{labeling_type}_{loss_type}_{nb_val}_{dropout}_{model_idx}.npy",
            {"GE": ge, "GE_rounds": ge_rounds, "NTGE": NTGE, "success_rate": success_rate, "accuracy": accuracy,
             "labeling_total_acc": labeling_total_acc, "labeling_acc_m": labeling_acc_m,
             "labeling_acc_y": labeling_acc_y})

