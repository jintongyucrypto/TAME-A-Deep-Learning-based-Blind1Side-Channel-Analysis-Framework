"""
LRE (Learning to Reweight Examples) for MLP and CNN, implemented in TensorFlow.

Reference: "Learning to Reweight Examples for Robust Deep Learning" (ICML 2018)

核心思想：用一个干净的小验证集，通过元学习自动为每个带噪训练样本计算权重。
  - 对验证集有利的训练样本 → 权重大
  - 对验证集有害的噪声样本 → 权重趋近于 0

关键步骤：
  1. 元网络初始化（复制主网络当前权重）
  2. 初始前向传播，计算加权损失（eps=0）
  3. 通过 inner GradientTape 模拟元网络更新一步
  4. 用更新后的元网络在验证集上前向，得到验证集损失
  5. 通过 outer GradientTape 对 eps 求梯度 → 得到每个样本的重要性信号
  6. 归一化权重，用归一化权重训练主网络
"""

import gc
import pickle

import numpy as np
import tensorflow as tf

def _act(name):
    return {'relu': tf.nn.relu, 'selu': tf.nn.selu,
            'elu': tf.nn.elu,  'tanh': tf.nn.tanh}[name]

class FunctionalMLP:
    """
    MLP whose weights are stored as tf.Variables.
    Supports forward(x, weights=...) with externally provided weight tensors,
    which is required for the meta-learning step in LRE.

    Weight layout in trainable_variables (and in any external weights list):
      [bn_gamma, bn_beta,
       W0, b0, W1, b1, ..., W_{L-1}, b_{L-1},   # hidden layers
       W_out, b_out]                               # output layer
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,
                 activation='relu', seed=42):
        tf.random.set_seed(seed)
        self.activation  = activation
        self.num_layers  = num_layers
        self.input_dim   = input_dim

        init = tf.keras.initializers.HeUniform(seed=seed)

        self.bn_gamma = tf.Variable(tf.ones([input_dim]),  trainable=True,  dtype=tf.float32, name='bn_gamma')
        self.bn_beta  = tf.Variable(tf.zeros([input_dim]), trainable=True,  dtype=tf.float32, name='bn_beta')
        self.bn_mean  = tf.Variable(tf.zeros([input_dim]), trainable=False, dtype=tf.float32, name='bn_mean')
        self.bn_var   = tf.Variable(tf.ones([input_dim]),  trainable=False, dtype=tf.float32, name='bn_var')

        self._dense_vars = []
        in_dim = input_dim
        for i in range(num_layers):
            W = tf.Variable(init(shape=[in_dim, hidden_dim]), trainable=True, dtype=tf.float32, name=f'W{i}')
            b = tf.Variable(tf.zeros([hidden_dim]),            trainable=True, dtype=tf.float32, name=f'b{i}')
            self._dense_vars.append((W, b))
            in_dim = hidden_dim

        self.W_out = tf.Variable(init(shape=[in_dim, output_dim]), trainable=True, dtype=tf.float32, name='W_out')
        self.b_out = tf.Variable(tf.zeros([output_dim]),            trainable=True, dtype=tf.float32, name='b_out')

    @property
    def trainable_variables(self):
        """Flat list of all trainable tf.Variables, in canonical order."""
        tvars = [self.bn_gamma, self.bn_beta]
        for W, b in self._dense_vars:
            tvars.extend([W, b])
        tvars.extend([self.W_out, self.b_out])
        return tvars

    def forward(self, x, weights=None, training=True):
        """
        Forward pass.

        Args:
            x        : input tensor, shape (N, input_dim)
            weights  : optional list of tensors in the same order as
                       trainable_variables.  When provided, those tensors are
                       used instead of self's variables (meta-learning step).
            training : if True use batch statistics for BN; else use running stats.
        Returns:
            logits, shape (N, classes)  — softmax NOT applied here.
        """
        if weights is None:
            bn_gamma   = self.bn_gamma
            bn_beta    = self.bn_beta
            dense_list = self._dense_vars
            W_out      = self.W_out
            b_out      = self.b_out
        else:
            bn_gamma = weights[0]
            bn_beta  = weights[1]
            idx = 2
            dense_list = []
            for _ in range(self.num_layers):
                dense_list.append((weights[idx], weights[idx + 1]))
                idx += 2
            W_out = weights[idx]
            b_out = weights[idx + 1]

        x = tf.cast(x, tf.float32)

        if training:
            batch_mean, batch_var = tf.nn.moments(x, axes=[0])
            x_norm = (x - batch_mean) / tf.sqrt(batch_var + 1e-5)
            if weights is None:
                self.bn_mean.assign(0.9 * self.bn_mean + 0.1 * batch_mean)
                self.bn_var.assign(0.9 * self.bn_var  + 0.1 * batch_var)
        else:
            x_norm = (x - self.bn_mean) / tf.sqrt(self.bn_var + 1e-5)

        x = x_norm * bn_gamma + bn_beta

        act_fn = _act(self.activation)
        for W, b in dense_list:
            x = act_fn(tf.matmul(x, W) + b)

        return tf.matmul(x, W_out) + b_out

    def get_weights_snapshot(self):
        """Return current trainable weights as a list of numpy arrays."""
        return [v.numpy() for v in self.trainable_variables]

    def set_weights_from_list(self, weight_list):
        """Restore weights from a list of numpy arrays."""
        for var, val in zip(self.trainable_variables, weight_list):
            var.assign(val)

def run_mlp_lre(x_profiling, x_attack, y_profiling_noisy,
                x_val, y_val_clean,
                actual_joint_train, actual_joint_test,
                epochs, classes,
                trained_folder, base, model_ind,
                dropout, loss_type, g_or_l, dataset):
    """
    Train an MLP with the LRE algorithm and return test-set predictions.

    Args:
        x_profiling       : (N_train, F) standardised training traces
        x_attack          : (N_test,  F) standardised test traces
        y_profiling_noisy : (N_train,)   noisy labels from Phase-2 labelling
        x_val             : (N_val, F)   clean validation traces (subset of training set)
        y_val_clean       : (N_val,)     true joint-HW labels for the validation set
        actual_joint_train: (N_train,)   true labels (only for accuracy evaluation)
        actual_joint_test : (N_test,)    true labels (only for accuracy evaluation)
        epochs            : number of training epochs
        classes           : number of output classes (729 for Ascon)
        trained_folder    : directory for saving trained model weights
        base              : directory for saving hyper-parameters and initial weights
        model_ind         : model index (0-99 for random search)
        dropout           : (unused in LRE variant, kept for API compatibility)
        loss_type         : "CCE" (kept for API compatibility)
        g_or_l            : "generate" | "load" | "load_trained"
        dataset           : dataset name string

    Returns:
        predicted_hw_t : (N_test,) predicted joint-HW for test traces
        accuracy       : dict with profile/attack accuracy metrics
    """
    import random
    from src.utils import customized_accuracy_ascon, customized_accuracy

    num_samples = x_profiling.shape[1]
    n_train     = x_profiling.shape[0]

    with open(f'{base}/Model_{model_ind}.pkl', 'rb') as f:
        param = pickle.load(f)

    model = FunctionalMLP(
        input_dim   = num_samples,
        hidden_dim  = param["neurons"],
        num_layers  = param["layers"],
        output_dim  = classes,
        activation  = param["activation"],
        seed        = param["seed"],
    )

    lr         = param["learning_rate"]
    mini_batch = param["mini_batch"]
    optimizer  = tf.keras.optimizers.Adam(learning_rate=lr)

    if g_or_l == "load_trained":
        print(f"[LRE] Loading trained model {model_ind}")
        with open(f'{trained_folder}trained_models/LRE_{loss_type}_{dropout}_{model_ind}_weights.pkl', 'rb') as f:
            model.set_weights_from_list(pickle.load(f))
        return _lre_predict_and_accuracy(
            model, x_profiling, x_attack,
            actual_joint_train, actual_joint_test, dataset, classes)

    init_path = f'{base}/initial_weights_{model_ind}.pkl'
    if g_or_l == "generate":
        with open(init_path, 'wb') as f:
            pickle.dump(model.get_weights_snapshot(), f)
    else:
        print("[LRE] Loading a good model (reproducible initial weights)!")
        with open(init_path, 'rb') as f:
            saved_weights = pickle.load(f)
        n_expected = len(model.trainable_variables)
        if len(saved_weights) == n_expected + 2:
            saved_weights = saved_weights[:2] + saved_weights[4:]
        model.set_weights_from_list(saved_weights)

    X_tr_np  = x_profiling.astype(np.float32)
    Y_tr_np  = y_profiling_noisy.astype(np.int32)
    X_val_tf = tf.constant(x_val,             dtype=tf.float32)
    Y_val_oh = tf.one_hot(
        tf.constant(y_val_clean, dtype=tf.int32), classes)

    num_iterations = epochs * max(1, n_train // mini_batch)
    print(f"[LRE] Training: iters={num_iterations}, batch={mini_batch}, "
          f"lr={lr}, layers={param['layers']}, neurons={param['neurons']}")

    for iteration in range(num_iterations):

        batch_idx  = np.random.choice(n_train, mini_batch, replace=False)
        x_batch    = tf.constant(X_tr_np[batch_idx], dtype=tf.float32)
        y_batch_oh = tf.one_hot(tf.constant(Y_tr_np[batch_idx], dtype=tf.int32), classes)

        eps = tf.zeros([mini_batch], dtype=tf.float32)

        with tf.GradientTape() as outer_tape:
            outer_tape.watch(eps)

            current_weights = [tf.identity(v) for v in model.trainable_variables]

            with tf.GradientTape() as inner_tape:
                inner_tape.watch(current_weights)

                logits_meta    = model.forward(x_batch, weights=current_weights, training=True)
                per_sample_loss = tf.keras.losses.categorical_crossentropy(
                    y_batch_oh, tf.nn.softmax(logits_meta))
                l_f_meta = tf.reduce_sum(per_sample_loss * eps)

            meta_grads = inner_tape.gradient(l_f_meta, current_weights)
            meta_grads = [g if g is not None else tf.zeros_like(w)
                          for g, w in zip(meta_grads, current_weights)]
            updated_weights = [w - lr * g for w, g in zip(current_weights, meta_grads)]

            logits_val = model.forward(X_val_tf, weights=updated_weights, training=True)
            l_g_meta   = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    Y_val_oh, tf.nn.softmax(logits_val)))

        grad_eps = outer_tape.gradient(l_g_meta, eps)

        if grad_eps is not None:
            w_tilde = tf.maximum(-grad_eps, 0.0)
            norm_c  = tf.reduce_sum(w_tilde)
            w_norm  = w_tilde / norm_c if norm_c > 1e-8 else tf.fill([mini_batch], 1.0 / mini_batch)
        else:
            w_norm = tf.fill([mini_batch], 1.0 / mini_batch)

        with tf.GradientTape() as train_tape:
            logits_train    = model.forward(x_batch, training=True)
            per_sample_loss = tf.keras.losses.categorical_crossentropy(
                y_batch_oh, tf.nn.softmax(logits_train))
            l_f = tf.reduce_sum(per_sample_loss * w_norm)

        grads = train_tape.gradient(l_f, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if iteration % 500 == 0:
            logits_v = model.forward(X_val_tf, training=False)
            val_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    Y_val_oh, tf.nn.softmax(logits_v)))
            nonzero = tf.reduce_sum(tf.cast(w_norm > 1e-6, tf.float32)).numpy()
            print(f"  iter {iteration:5d}/{num_iterations} | "
                  f"val_loss={val_loss:.4f} | "
                  f"nonzero_weights={int(nonzero)}/{mini_batch}")

    save_path = f"{trained_folder}trained_models/LRE_{loss_type}_{dropout}_{model_ind}_weights.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(model.get_weights_snapshot(), f)
    print(f"[LRE] Saved trained weights → {save_path}")

    predicted_hw_t, accuracy = _lre_predict_and_accuracy(
        model, x_profiling, x_attack,
        actual_joint_train, actual_joint_test, dataset, classes)

    del model
    gc.collect()
    return predicted_hw_t, accuracy

def run_mlp_lre_v2(x_profiling, x_attack, y_profiling_noisy,
                   x_val, y_val_clean,
                   actual_joint_train, actual_joint_test,
                   epochs, classes,
                   trained_folder, base, model_ind,
                   dropout, loss_type, g_or_l, dataset):
    """
    与 run_mlp_lre 接口完全相同，包含以下修复：
      - Fix-1: norm_c=0 时使用零权重（跳过更新），而非均匀权重
      - Fix-2: 所有 CCE 损失改用 from_logits=True（数值稳定）
    """
    import random
    from src.utils import customized_accuracy_ascon, customized_accuracy

    num_samples = x_profiling.shape[1]
    n_train     = x_profiling.shape[0]

    with open(f'{base}/Model_{model_ind}.pkl', 'rb') as f:
        param = pickle.load(f)

    model = FunctionalMLP(
        input_dim   = num_samples,
        hidden_dim  = param["neurons"],
        num_layers  = param["layers"],
        output_dim  = classes,
        activation  = param["activation"],
        seed        = param["seed"],
    )

    lr         = param["learning_rate"]
    mini_batch = param["mini_batch"]
    optimizer  = tf.keras.optimizers.Adam(learning_rate=lr)

    if g_or_l == "load_trained":
        print(f"[LRE-v2] Loading trained model {model_ind}")
        with open(f'{trained_folder}trained_models/LRE_{loss_type}_{dropout}_{model_ind}_weights.pkl', 'rb') as f:
            model.set_weights_from_list(pickle.load(f))
        return _lre_predict_and_accuracy(
            model, x_profiling, x_attack,
            actual_joint_train, actual_joint_test, dataset, classes)

    init_path = f'{base}/initial_weights_{model_ind}.pkl'
    if g_or_l == "generate":
        with open(init_path, 'wb') as f:
            pickle.dump(model.get_weights_snapshot(), f)
    else:
        print("[LRE-v2] Loading a good model (reproducible initial weights)!")
        with open(init_path, 'rb') as f:
            model.set_weights_from_list(pickle.load(f))

    X_tr_np  = x_profiling.astype(np.float32)
    Y_tr_np  = y_profiling_noisy.astype(np.int32)
    X_val_tf = tf.constant(x_val,             dtype=tf.float32)
    Y_val_oh = tf.one_hot(
        tf.constant(y_val_clean, dtype=tf.int32), classes)

    num_iterations = epochs * max(1, n_train // mini_batch)
    print(f"[LRE-v2] Training: iters={num_iterations}, batch={mini_batch}, "
          f"lr={lr}, layers={param['layers']}, neurons={param['neurons']}")

    for iteration in range(num_iterations):

        batch_idx  = np.random.choice(n_train, mini_batch, replace=False)
        x_batch    = tf.constant(X_tr_np[batch_idx], dtype=tf.float32)
        y_batch_oh = tf.one_hot(tf.constant(Y_tr_np[batch_idx], dtype=tf.int32), classes)

        eps = tf.zeros([mini_batch], dtype=tf.float32)

        with tf.GradientTape() as outer_tape:
            outer_tape.watch(eps)

            current_weights = [tf.identity(v) for v in model.trainable_variables]

            with tf.GradientTape() as inner_tape:
                inner_tape.watch(current_weights)
                logits_meta     = model.forward(x_batch, weights=current_weights, training=True)
                per_sample_loss = tf.keras.losses.categorical_crossentropy(
                    y_batch_oh, logits_meta, from_logits=True)
                l_f_meta = tf.reduce_sum(per_sample_loss * eps)

            meta_grads = inner_tape.gradient(l_f_meta, current_weights)
            meta_grads = [g if g is not None else tf.zeros_like(w)
                          for g, w in zip(meta_grads, current_weights)]
            updated_weights = [w - lr * g for w, g in zip(current_weights, meta_grads)]

            logits_val = model.forward(X_val_tf, weights=updated_weights, training=True)
            l_g_meta   = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    Y_val_oh, logits_val, from_logits=True))

        grad_eps = outer_tape.gradient(l_g_meta, eps)

        if grad_eps is not None:
            w_tilde = tf.maximum(-grad_eps, 0.0)
            norm_c  = tf.reduce_sum(w_tilde)
            w_norm  = tf.cond(tf.greater(norm_c, 1e-8), lambda: w_tilde / norm_c, lambda: tf.zeros([mini_batch]))
        else:
            w_norm = tf.zeros([mini_batch])

        with tf.GradientTape() as train_tape:
            logits_train    = model.forward(x_batch, training=True)
            per_sample_loss = tf.keras.losses.categorical_crossentropy(
                y_batch_oh, logits_train, from_logits=True)
            l_f = tf.reduce_sum(per_sample_loss * w_norm)

        grads = train_tape.gradient(l_f, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if iteration % 500 == 0:
            logits_v = model.forward(X_val_tf, training=False)
            val_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    Y_val_oh, logits_v, from_logits=True))
            nonzero = tf.reduce_sum(tf.cast(w_norm > 1e-6, tf.float32)).numpy()
            print(f"  iter {iteration:5d}/{num_iterations} | "
                  f"val_loss={val_loss:.4f} | "
                  f"nonzero_weights={int(nonzero)}/{mini_batch}")

    save_path = f"{trained_folder}trained_models/LRE_{loss_type}_{dropout}_{model_ind}_weights.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(model.get_weights_snapshot(), f)
    print(f"[LRE-v2] Saved trained weights → {save_path}")

    predicted_hw_t, accuracy = _lre_predict_and_accuracy(
        model, x_profiling, x_attack,
        actual_joint_train, actual_joint_test, dataset, classes)

    del model
    gc.collect()
    return predicted_hw_t, accuracy

def run_cnn_lre_v2(x_profiling, x_attack, y_profiling_noisy,
                   x_val, y_val_clean,
                   actual_joint_train, actual_joint_test,
                   epochs, classes,
                   trained_folder, base, model_ind,
                   dropout, loss_type, g_or_l, dataset):
    """
    与 run_cnn_lre 接口完全相同，包含以下修复：
      - Fix-1: norm_c=0 时使用零权重（跳过更新），而非均匀权重
      - Fix-2: 所有 CCE 损失改用 from_logits=True（数值稳定）
    """
    from src.utils import customized_accuracy_ascon, customized_accuracy

    input_len = x_profiling.shape[1]
    n_train   = x_profiling.shape[0]

    if g_or_l == "generate":
        param = cnn_network_parameters_lre()
        with open(f'{base}/CNN_Model_{model_ind}.pkl', 'wb') as f:
            pickle.dump(param, f)
    elif g_or_l in ("load", "load_trained"):
        with open(f'{base}/CNN_Model_{model_ind}.pkl', 'rb') as f:
            param = pickle.load(f)

    model      = FunctionalCNN(input_len, param, output_dim=classes)
    lr         = param["learning_rate"]
    mini_batch = param["mini_batch"]
    optimizer  = tf.keras.optimizers.Adam(learning_rate=lr)

    if g_or_l == "load_trained":
        print(f"[LRE-CNN-v2] Loading trained model {model_ind}")
        wpath = (f'{trained_folder}trained_models/'
                 f'LRE_CNN_{loss_type}_{dropout}_{model_ind}_weights.pkl')
        with open(wpath, 'rb') as f:
            model.set_weights_from_list(pickle.load(f))
        return _lre_predict_and_accuracy(
            model, x_profiling, x_attack,
            actual_joint_train, actual_joint_test, dataset, classes)

    init_path = f'{base}/CNN_initial_weights_{model_ind}.pkl'
    if g_or_l == "generate":
        with open(init_path, 'wb') as f:
            pickle.dump(model.get_weights_snapshot(), f)
    else:
        print("[LRE-CNN-v2] Loading a good model (reproducible initial weights)!")
        with open(init_path, 'rb') as f:
            model.set_weights_from_list(pickle.load(f))

    X_tr_np  = x_profiling.astype(np.float32)
    Y_tr_np  = y_profiling_noisy.astype(np.int32)
    X_val_tf = tf.constant(x_val,             dtype=tf.float32)
    Y_val_oh = tf.one_hot(
        tf.constant(y_val_clean, dtype=tf.int32), classes)

    num_iterations = epochs * max(1, n_train // mini_batch)
    print(f"[LRE-CNN-v2] Training: iters={num_iterations}, batch={mini_batch}, lr={lr}, "
          f"conv={param['conv_layers']}, dense={param['dense_layers']}, "
          f"neurons={param['neurons']}, flat_dim={model.flat_dim}")

    for iteration in range(num_iterations):

        batch_idx  = np.random.choice(n_train, mini_batch, replace=False)
        x_batch    = tf.constant(X_tr_np[batch_idx], dtype=tf.float32)
        y_batch_oh = tf.one_hot(tf.constant(Y_tr_np[batch_idx], dtype=tf.int32), classes)

        eps = tf.zeros([mini_batch], dtype=tf.float32)

        with tf.GradientTape() as outer_tape:
            outer_tape.watch(eps)

            current_weights = [tf.identity(v) for v in model.trainable_variables]

            with tf.GradientTape() as inner_tape:
                inner_tape.watch(current_weights)
                logits_meta     = model.forward(x_batch, weights=current_weights, training=True)
                per_sample_loss = tf.keras.losses.categorical_crossentropy(
                    y_batch_oh, logits_meta, from_logits=True)
                l_f_meta = tf.reduce_sum(per_sample_loss * eps)

            meta_grads = inner_tape.gradient(l_f_meta, current_weights)
            meta_grads = [g if g is not None else tf.zeros_like(w)
                          for g, w in zip(meta_grads, current_weights)]
            updated_weights = [w - lr * g for w, g in zip(current_weights, meta_grads)]

            logits_val = model.forward(X_val_tf, weights=updated_weights, training=True)
            l_g_meta   = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    Y_val_oh, logits_val, from_logits=True))

        grad_eps = outer_tape.gradient(l_g_meta, eps)

        if grad_eps is not None:
            w_tilde = tf.maximum(-grad_eps, 0.0)
            norm_c  = tf.reduce_sum(w_tilde)
            w_norm  = tf.cond(tf.greater(norm_c, 1e-8), lambda: w_tilde / norm_c, lambda: tf.zeros([mini_batch]))
        else:
            w_norm = tf.zeros([mini_batch])

        with tf.GradientTape() as train_tape:
            logits_train    = model.forward(x_batch, training=True)
            per_sample_loss = tf.keras.losses.categorical_crossentropy(
                y_batch_oh, logits_train, from_logits=True)
            l_f = tf.reduce_sum(per_sample_loss * w_norm)

        grads = train_tape.gradient(l_f, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if iteration % 500 == 0:
            logits_v = model.forward(X_val_tf, training=False)
            val_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    Y_val_oh, logits_v, from_logits=True))
            nonzero = tf.reduce_sum(tf.cast(w_norm > 1e-6, tf.float32)).numpy()
            print(f"  iter {iteration:5d}/{num_iterations} | "
                  f"val_loss={val_loss:.4f} | "
                  f"nonzero_weights={int(nonzero)}/{mini_batch}")

    save_path = (f"{trained_folder}trained_models/"
                 f"LRE_CNN_{loss_type}_{dropout}_{model_ind}_weights.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(model.get_weights_snapshot(), f)
    print(f"[LRE-CNN-v2] Saved trained weights → {save_path}")

    predicted_hw_t, accuracy = _lre_predict_and_accuracy(
        model, x_profiling, x_attack,
        actual_joint_train, actual_joint_test, dataset, classes)

    del model
    gc.collect()
    return predicted_hw_t, accuracy

def _lre_predict_and_accuracy(model, x_profiling, x_attack,
                               actual_joint_train, actual_joint_test,
                               dataset, classes):
    from src.utils import customized_accuracy_ascon, customized_accuracy

    PRED_BATCH = 1024

    def predict_batched(x_np):
        parts = []
        x_np = np.asarray(x_np, dtype=np.float32)
        for start in range(0, len(x_np), PRED_BATCH):
            x_b = tf.constant(x_np[start:start + PRED_BATCH], dtype=tf.float32)
            parts.append(np.argmax(model.forward(x_b, training=False).numpy(), axis=1))
        return np.concatenate(parts)

    predicted_hw_p = predict_batched(x_profiling)
    predicted_hw_t = predict_batched(x_attack)

    if dataset == "Ascon":
        p_tot, p_m1, p_m2, p_y, p_sm1, p_sm2, p_sy = \
            customized_accuracy_ascon(predicted_hw_p, actual_joint_train)
        a_tot, a_m1, a_m2, a_y, a_sm1, a_sm2, a_sy = \
            customized_accuracy_ascon(predicted_hw_t, actual_joint_test)
        accuracy = {
            'profile_acc':    p_tot,
            'profile_acc_m1': p_m1,  'profile_acc_m2': p_m2,  'profile_acc_y': p_y,
            'profile_acc_single_m1': p_sm1, 'profile_acc_single_m2': p_sm2, 'profile_acc_single_y': p_sy,
            'attack_acc':     a_tot,
            'attack_acc_m1':  a_m1,  'attack_acc_m2':  a_m2,  'attack_acc_y':  a_y,
            'attack_acc_single_m1': a_sm1, 'attack_acc_single_m2': a_sm2, 'attack_acc_single_y': a_sy,
        }
    else:
        p_tot, p_m, p_y = customized_accuracy(predicted_hw_p, actual_joint_train, classes)
        a_tot, a_m, a_y = customized_accuracy(predicted_hw_t, actual_joint_test, classes)
        accuracy = {
            'profile_acc': p_tot, 'profile_acc_m': p_m, 'profile_acc_y': p_y,
            'attack_acc':  a_tot, 'attack_acc_m':  a_m, 'attack_acc_y':  a_y,
        }

    return predicted_hw_t, accuracy

def cnn_network_parameters_lre():
    """
    与 neural_networks.cnn_network_parameters 结构相同，
    但 pooling 以普通 dict 存储（可 pickle），而非 Keras 层对象。
    """
    import random
    param = {}
    param["seed"]          = int(np.random.randint(1048576))
    param["mini_batch"]    = random.choice([128, 256, 512])
    param["learning_rate"] = random.choice([1e-3, 5e-4, 1e-4, 5e-5])
    param["activation"]    = random.choice(["relu", "selu", "elu", "tanh"])
    param["dense_layers"]  = random.randrange(2, 4, 1)
    param["neurons"]       = random.choice([50, 100, 150, 200, 300, 400, 500])
    param["conv_layers"]   = random.choice([2, 3, 4])
    param["pooling_type"]  = random.choice(["Average", "Max"])

    param["kernel_size"]     = []
    param["strides"]         = []
    param["filters"]         = []
    param["pooling_sizes"]   = []
    param["pooling_strides"] = []

    for i in range(1, param["conv_layers"] + 1):
        param["kernel_size"].append(random.randrange(4, 20, 2))
        param["filters"].append(
            random.choice([4, 8, 12, 16, 24]) if i == 1
            else param["filters"][i - 2] * 2
        )
        param["strides"].append(random.choice([2, 4, 6, 8, 10]))
        param["pooling_sizes"].append(random.choice([4, 6, 8, 10]))
        param["pooling_strides"].append(random.choice([4, 6, 8, 10]))

    return param

class FunctionalCNN:
    """
    CNN whose weights are stored as tf.Variables.
    Supports forward(x, weights=...) for the LRE meta-learning step.

    Architecture per conv block:  Conv1D → Pool1D → BatchNorm
    Followed by:  Flatten → Dense × dense_layers → Dense(classes)

    Trainable weight layout (same order as trainable_variables):
      Per conv block i  : [conv_kernel_i, conv_bias_i, bn_gamma_i, bn_beta_i]
      Per dense layer j : [W_j, b_j]
      Output layer      : [W_out, b_out]

    Non-trainable (BN running stats, NOT in trainable_variables):
      _bn_means[i], _bn_vars[i]
    """

    def __init__(self, input_len, param, output_dim):
        tf.random.set_seed(param["seed"])
        self.param      = param
        self.output_dim = output_dim
        self.activation = param["activation"]

        conv_layers  = param["conv_layers"]
        dense_layers = param["dense_layers"]
        neurons      = param["neurons"]
        init = tf.keras.initializers.HeUniform(seed=param["seed"])

        self._conv_kernels = []
        self._conv_biases  = []
        self._bn_gammas    = []
        self._bn_betas     = []
        self._bn_means     = []
        self._bn_vars      = []

        in_ch = 1
        for i in range(conv_layers):
            out_ch = param["filters"][i]
            k_size = param["kernel_size"][i]
            self._conv_kernels.append(
                tf.Variable(init(shape=[k_size, in_ch, out_ch]),
                            trainable=True, dtype=tf.float32, name=f'ck{i}'))
            self._conv_biases.append(
                tf.Variable(tf.zeros([out_ch]),
                            trainable=True, dtype=tf.float32, name=f'cb{i}'))
            self._bn_gammas.append(
                tf.Variable(tf.ones([out_ch]),
                            trainable=True, dtype=tf.float32, name=f'bg{i}'))
            self._bn_betas.append(
                tf.Variable(tf.zeros([out_ch]),
                            trainable=True, dtype=tf.float32, name=f'bb{i}'))
            self._bn_means.append(
                tf.Variable(tf.zeros([out_ch]),
                            trainable=False, dtype=tf.float32, name=f'bm{i}'))
            self._bn_vars.append(
                tf.Variable(tf.ones([out_ch]),
                            trainable=False, dtype=tf.float32, name=f'bv{i}'))
            in_ch = out_ch

        self.flat_dim = self._compute_flat_dim(tf.zeros([1, input_len, 1]))

        self._dense_W = []
        self._dense_b = []
        in_d = self.flat_dim
        for j in range(dense_layers):
            self._dense_W.append(
                tf.Variable(init(shape=[in_d, neurons]),
                            trainable=True, dtype=tf.float32, name=f'dW{j}'))
            self._dense_b.append(
                tf.Variable(tf.zeros([neurons]),
                            trainable=True, dtype=tf.float32, name=f'db{j}'))
            in_d = neurons

        self.W_out = tf.Variable(init(shape=[in_d, output_dim]),
                                 trainable=True, dtype=tf.float32, name='W_out')
        self.b_out = tf.Variable(tf.zeros([output_dim]),
                                 trainable=True, dtype=tf.float32, name='b_out')

    def _compute_flat_dim(self, x):
        """Pass a zero-filled dummy tensor through conv blocks to measure output size."""
        for i in range(self.param["conv_layers"]):
            k_size = self.param["kernel_size"][i]
            out_ch = self.param["filters"][i]
            stride = self.param["strides"][i]
            ps     = self.param["pooling_sizes"][i]
            pst    = self.param["pooling_strides"][i]
            x = tf.nn.conv1d(x, tf.zeros([k_size, x.shape[-1], out_ch]),
                             stride=stride, padding='SAME')
            x = tf.nn.avg_pool1d(x, ksize=ps, strides=pst, padding='SAME')
        return int(x.shape[1]) * int(x.shape[2])

    @property
    def trainable_variables(self):
        tvars = []
        for i in range(self.param["conv_layers"]):
            tvars.extend([self._conv_kernels[i], self._conv_biases[i],
                          self._bn_gammas[i],    self._bn_betas[i]])
        for j in range(self.param["dense_layers"]):
            tvars.extend([self._dense_W[j], self._dense_b[j]])
        tvars.extend([self.W_out, self.b_out])
        return tvars

    def forward(self, x, weights=None, training=True):
        """
        Forward pass.

        Args:
            x        : (N, W) or (N, W, 1) — reshaped to (N, W, 1) internally.
            weights  : optional external weight tensors in the same canonical
                       order as trainable_variables (for meta-learning).
            training : use batch stats for BN if True; running stats if False.
        Returns:
            logits, shape (N, classes) — softmax NOT applied.
        """
        conv_layers  = self.param["conv_layers"]
        dense_layers = self.param["dense_layers"]
        act_fn       = _act(self.activation)

        if weights is None:
            ck, cb = self._conv_kernels, self._conv_biases
            bg, bb = self._bn_gammas,   self._bn_betas
            dW, db = self._dense_W,     self._dense_b
            W_out, b_out = self.W_out,  self.b_out
        else:
            idx = 0
            ck, cb, bg, bb = [], [], [], []
            for _ in range(conv_layers):
                ck.append(weights[idx]);     cb.append(weights[idx + 1])
                bg.append(weights[idx + 2]); bb.append(weights[idx + 3])
                idx += 4
            dW, db = [], []
            for _ in range(dense_layers):
                dW.append(weights[idx]);  db.append(weights[idx + 1])
                idx += 2
            W_out = weights[idx];  b_out = weights[idx + 1]

        x = tf.cast(x, tf.float32)
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=-1)

        for i in range(conv_layers):
            x = tf.nn.conv1d(x, ck[i],
                             stride=self.param["strides"][i],
                             padding='SAME') + cb[i]
            ps  = self.param["pooling_sizes"][i]
            pst = self.param["pooling_strides"][i]
            if self.param["pooling_type"] == "Average":
                x = tf.nn.avg_pool1d(x, ksize=ps, strides=pst, padding='SAME')
            else:
                x = tf.nn.max_pool1d(x, ksize=ps, strides=pst, padding='SAME')
            if training:
                mean, var = tf.nn.moments(x, axes=[0, 1])
                x = (x - mean) / tf.sqrt(var + 1e-5)
                if weights is None:
                    self._bn_means[i].assign(0.9 * self._bn_means[i] + 0.1 * mean)
                    self._bn_vars[i].assign( 0.9 * self._bn_vars[i]  + 0.1 * var)
            else:
                x = (x - self._bn_means[i]) / tf.sqrt(self._bn_vars[i] + 1e-5)
            x = x * bg[i] + bb[i]

        x = tf.reshape(x, [tf.shape(x)[0], -1])

        for j in range(dense_layers):
            x = act_fn(tf.matmul(x, dW[j]) + db[j])

        return tf.matmul(x, W_out) + b_out

    def get_weights_snapshot(self):
        return [v.numpy() for v in self.trainable_variables]

    def set_weights_from_list(self, weight_list):
        for var, val in zip(self.trainable_variables, weight_list):
            var.assign(val)

def run_cnn_lre(x_profiling, x_attack, y_profiling_noisy,
                x_val, y_val_clean,
                actual_joint_train, actual_joint_test,
                epochs, classes,
                trained_folder, base, model_ind,
                dropout, loss_type, g_or_l, dataset):
    """
    Train a CNN with the LRE algorithm and return test-set predictions.

    Interface mirrors run_mlp_lre exactly.
    x_profiling / x_attack / x_val shape: (N, W) — channel dim added internally.
    """
    from src.utils import customized_accuracy_ascon, customized_accuracy

    input_len = x_profiling.shape[1]
    n_train   = x_profiling.shape[0]

    if g_or_l == "generate":
        param = cnn_network_parameters_lre()
        with open(f'{base}/CNN_Model_{model_ind}.pkl', 'wb') as f:
            pickle.dump(param, f)
    elif g_or_l in ("load", "load_trained"):
        with open(f'{base}/Model_{model_ind}.pkl', 'rb') as f:
            param = pickle.load(f)

    model      = FunctionalCNN(input_len, param, output_dim=classes)
    lr         = param["learning_rate"]
    mini_batch = param["mini_batch"]
    optimizer  = tf.keras.optimizers.Adam(learning_rate=lr)

    if g_or_l == "load_trained":
        print(f"[LRE-CNN] Loading trained model {model_ind}")
        wpath = (f'{trained_folder}trained_models/'
                 f'LRE_CNN_{loss_type}_{dropout}_{model_ind}_weights.pkl')
        with open(wpath, 'rb') as f:
            model.set_weights_from_list(pickle.load(f))
        return _lre_predict_and_accuracy(
            model, x_profiling, x_attack,
            actual_joint_train, actual_joint_test, dataset, classes)

    init_path = f'{base}/initial_weights_{model_ind}.pkl'
    if g_or_l == "generate":
        with open(init_path, 'wb') as f:
            pickle.dump(model.get_weights_snapshot(), f)
    else:
        print("[LRE-CNN] Loading a good model (reproducible initial weights)!")
        with open(init_path, 'rb') as f:
            saved_weights = pickle.load(f)
        n_conv    = param["conv_layers"]
        n_dense   = param["dense_layers"]
        keras_len = 6 * n_conv + 2 * n_dense + 2
        if len(saved_weights) == keras_len:
            converted = []
            idx = 0
            for _ in range(n_conv):
                converted.extend(saved_weights[idx:idx + 4])
                idx += 6
            converted.extend(saved_weights[idx:])
            saved_weights = converted
        model.set_weights_from_list(saved_weights)

    X_tr     = tf.constant(x_profiling,       dtype=tf.float32)
    Y_tr     = tf.constant(y_profiling_noisy, dtype=tf.int32)
    X_val_tf = tf.constant(x_val,             dtype=tf.float32)
    Y_val_oh = tf.one_hot(
        tf.constant(y_val_clean, dtype=tf.int32), classes)

    num_iterations = epochs * max(1, n_train // mini_batch)
    print(f"[LRE-CNN] Training: iters={num_iterations}, batch={mini_batch}, lr={lr}, "
          f"conv={param['conv_layers']}, dense={param['dense_layers']}, "
          f"neurons={param['neurons']}, flat_dim={model.flat_dim}")

    for iteration in range(num_iterations):

        batch_idx  = np.random.choice(n_train, mini_batch, replace=False)
        x_batch    = tf.gather(X_tr, batch_idx)
        y_batch_oh = tf.one_hot(tf.gather(Y_tr, batch_idx), classes)

        eps = tf.zeros([mini_batch], dtype=tf.float32)

        with tf.GradientTape() as outer_tape:
            outer_tape.watch(eps)

            current_weights = [tf.identity(v) for v in model.trainable_variables]

            with tf.GradientTape() as inner_tape:
                inner_tape.watch(current_weights)
                logits_meta     = model.forward(x_batch, weights=current_weights, training=True)
                per_sample_loss = tf.keras.losses.categorical_crossentropy(
                    y_batch_oh, tf.nn.softmax(logits_meta))
                l_f_meta = tf.reduce_sum(per_sample_loss * eps)

            meta_grads = inner_tape.gradient(l_f_meta, current_weights)
            meta_grads = [g if g is not None else tf.zeros_like(w)
                          for g, w in zip(meta_grads, current_weights)]
            updated_weights = [w - lr * g for w, g in zip(current_weights, meta_grads)]

            logits_val = model.forward(X_val_tf, weights=updated_weights, training=True)
            l_g_meta   = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    Y_val_oh, tf.nn.softmax(logits_val)))

        grad_eps = outer_tape.gradient(l_g_meta, eps)

        if grad_eps is not None:
            w_tilde = tf.maximum(-grad_eps, 0.0)
            norm_c  = tf.reduce_sum(w_tilde)
            w_norm  = (w_tilde / norm_c if norm_c > 1e-8
                       else tf.fill([mini_batch], 1.0 / mini_batch))
        else:
            w_norm = tf.fill([mini_batch], 1.0 / mini_batch)

        with tf.GradientTape() as train_tape:
            logits_train    = model.forward(x_batch, training=True)
            per_sample_loss = tf.keras.losses.categorical_crossentropy(
                y_batch_oh, tf.nn.softmax(logits_train))
            l_f = tf.reduce_sum(per_sample_loss * w_norm)

        grads = train_tape.gradient(l_f, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if iteration % 500 == 0:
            logits_v = model.forward(X_val_tf, training=False)
            val_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    Y_val_oh, tf.nn.softmax(logits_v)))
            nonzero = tf.reduce_sum(tf.cast(w_norm > 1e-6, tf.float32)).numpy()
            print(f"  iter {iteration:5d}/{num_iterations} | "
                  f"val_loss={val_loss:.4f} | "
                  f"nonzero_weights={int(nonzero)}/{mini_batch}")

    save_path = (f"{trained_folder}trained_models/"
                 f"LRE_CNN_{loss_type}_{dropout}_{model_ind}_weights.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(model.get_weights_snapshot(), f)
    print(f"[LRE-CNN] Saved trained weights → {save_path}")

    predicted_hw_t, accuracy = _lre_predict_and_accuracy(
        model, x_profiling, x_attack,
        actual_joint_train, actual_joint_test, dataset, classes)

    del model
    gc.collect()
    return predicted_hw_t, accuracy
