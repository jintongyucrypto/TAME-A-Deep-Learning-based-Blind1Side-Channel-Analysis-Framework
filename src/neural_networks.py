from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout, GaussianNoise
from tensorflow.keras.layers import Conv1D, Conv2D, AveragePooling2D, MaxPooling1D, MaxPooling2D, BatchNormalization, AveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as backend
from tensorflow.keras.layers import MaxPool1D, AveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import gc
import numpy as np
import random
import pickle

import tensorflow as tf
from tensorflow.keras import backend as K
from src.utils import customized_accuracy, customized_accuracy_ascon

def mlp_network_parameters():
    param = {}
    param["mini_batch"] = random.choice([128, 256, 512])
    param["learning_rate"] = random.choice([1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
    param["activation"] = random.choice(["relu", "selu", "elu", "tanh"])
    param["kernel_initializer"] = random.choice(["random_uniform", "glorot_uniform", "he_uniform"])
    param["layers"] = random.randrange(2, 8, 1)
    param["neurons"] = random.choice([10, 30, 50, 70, 90, 120, 150, 200, 250, 300, 400, 500])
    param["seed"] = np.random.randint(1048576)
    return param

def mlp_random(classes, number_of_samples, m_params, dropout, loss_type):
    tf.random.set_seed(m_params["seed"])
    model = Sequential()
    model.add(BatchNormalization(input_shape=(number_of_samples,)))
    for l_i in range(m_params["layers"]):
        model.add(Dense(m_params["neurons"], activation=m_params["activation"], kernel_initializer='he_uniform',
                        bias_initializer='zeros'))
        if dropout:
            model.add(Dropout(0.2))
    model.add(Dense(classes, activation='softmax'))
    optimizer = Adam(learning_rate=m_params["learning_rate"])
    if loss_type == "PEER_LOSS":
        model.compile(loss=peer_DMI, optimizer=optimizer, metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def run_mlp(x_profiling, x_attack, y_profiling, actual_joint_train, actual_joint_test, epochs, classes, trained_folder,
            base, model_ind, dropout, loss_type, g_or_l, dataset):
    num_samples = x_profiling.shape[1]
    if g_or_l == "generate":
        mlp_params = mlp_network_parameters()
        with open(f'{base}/Model_{model_ind}.pkl', 'wb') as file:
            pickle.dump(mlp_params, file)

    elif g_or_l == "load":
        print("loading a good model!")
        with open(f'{base}/Model_{model_ind}.pkl', 'rb') as file:
            mlp_params = pickle.load(file)

    if g_or_l in ["generate", "load"]:
        mini_batch = mlp_params["mini_batch"]
        model = mlp_random(classes, num_samples, mlp_params, dropout, loss_type)
        if g_or_l == "generate":
            initial_weights = model.get_weights()
            with open(f'{base}/initial_weights_{model_ind}.pkl', 'wb') as f:
                pickle.dump(initial_weights, f)
        elif g_or_l == "load":
            with open(f'{base}/initial_weights_{model_ind}.pkl', 'rb') as f:
                initial_weights = pickle.load(f)
            model.set_weights(initial_weights)

        model.fit(
            x=x_profiling,
            y=to_categorical(y_profiling, num_classes=classes),
            batch_size=mini_batch,
            verbose=0,
            epochs=epochs,
            shuffle=True,
            callbacks=[],)
        model.save(f"{trained_folder}trained_models/{loss_type}_{dropout}_{model_ind}.h5")
    elif g_or_l == "load_trained":
        print(f'Loading model {model_ind}')
        model = load_model(trained_folder + f"trained_models/{loss_type}_{dropout}_{model_ind}.h5")
    else:
        print("Error!")
        exit(-1)
    predictions = np.array(model.predict(x_profiling))
    predicted_hw_p = np.argmax(predictions, axis=1)

    predictions = np.array(model.predict(x_attack))
    predicted_hw_t = np.argmax(predictions, axis=1)

    if dataset in ("Kyber", "Chipwhisperer"):
        profile_total_acc, profile_detailed_acc_m, profile_detailed_acc_y = \
            customized_accuracy(predicted_hw_p, actual_joint_train, classes)
        attack_total_acc, attack_detailed_acc_m, attack_detailed_acc_y = \
            customized_accuracy(predicted_hw_t, actual_joint_test, classes)
        accuracy = {
            'profile_acc_m': profile_detailed_acc_m,
            'profile_acc_y': profile_detailed_acc_y,
            'profile_acc': profile_total_acc,
            'attack_acc_m': attack_detailed_acc_m,
            'attack_acc_y': attack_detailed_acc_y,
            'attack_acc': attack_total_acc
        }
    else:
        profile_total_acc, profile_detailed_acc_m1, profile_detailed_acc_m2, profile_detailed_acc_y, profile_acc_m1, profile_acc_m2, profile_acc_y = \
            customized_accuracy_ascon(predicted_hw_p, actual_joint_train)
        attack_total_acc, attack_detailed_acc_m1, attack_detailed_acc_m2, attack_detailed_acc_y, attack_acc_m1, attack_acc_m2, attack_acc_y = \
            customized_accuracy_ascon(predicted_hw_t, actual_joint_test)
        accuracy = {
            'profile_acc_m1': profile_detailed_acc_m1,
            'profile_acc_m2': profile_detailed_acc_m2,
            'profile_acc_y': profile_detailed_acc_y,
            'profile_acc': profile_total_acc,
            'profile_acc_single_m1': profile_acc_m1,
            'profile_acc_single_m2': profile_acc_m2,
            'profile_acc_single_y': profile_acc_y,
            'attack_acc_m1': attack_detailed_acc_m1,
            'attack_acc_m2': attack_detailed_acc_m2,
            'attack_acc_y': attack_detailed_acc_y,
            'attack_acc': attack_total_acc,
            'attack_acc_single_m1': attack_acc_m1,
            'attack_acc_single_m2': attack_acc_m2,
            'attack_acc_single_y': attack_acc_y
        }

    del model
    backend.clear_session()
    gc.collect()

    return predicted_hw_t, accuracy

def cnn_network_parameters():
    param = {}
    param["seed"] = np.random.randint(1048576)
    param["mini_batch"] = random.choice([128, 256, 512])
    param["learning_rate"] = random.choice([1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
    param["activation"] = random.choice(["relu", "selu", "elu", "tanh"])
    param["kernel_initializer"] = random.choice(["random_uniform", "glorot_uniform", "he_uniform"])
    param["dense_layers"] = random.randrange(2, 4, 1)
    param["neurons"] = random.choice([50, 100, 150, 200, 300, 400, 500])
    param["conv_layers"] = random.choice([2, 3, 4])
    param["pooling_type"] = random.choice(["Average", "Max"])

    param["pooling_types"] = []
    param["pooling_layers"] = []
    param["pooling_sizes"] = []
    param["pooling_strides"] = []

    param["kernel_size"] = []
    param["strides"] = []
    param["filters"] = []
    for conv_layer in range(1, param["conv_layers"] + 1):
        param["kernel_size"].append(random.randrange(4, 20, 2))
        if conv_layer == 1:
            param["filters"].append(random.choice([4, 8, 12, 16, 24]))
        else:
            param["filters"].append(param["filters"][conv_layer - 2] * 2)
        param["strides"].append(random.choice([2, 4, 6, 8, 10]))
        param["pooling_sizes"].append(random.choice([4, 6, 8, 10]))
        param["pooling_strides"].append(random.choice([4, 6, 8, 10]))
        param["pooling_types"].append(param["pooling_type"])
        if param["pooling_types"][conv_layer - 1] == "Average":
            param["pooling_layers"].append(
                AveragePooling1D(pool_size=param["pooling_sizes"][conv_layer - 1],
                                 strides=param["pooling_strides"][conv_layer - 1],
                                 padding="same"))
        elif param["pooling_types"][conv_layer - 1] == "Max":
            param["pooling_layers"].append(
                MaxPool1D(pool_size=param["pooling_sizes"][conv_layer - 1],
                          strides=param["pooling_strides"][conv_layer - 1],
                          padding="same"))
    return param

def cnn_random(classes, number_of_samples, m_params, dropout, loss_type):
    model = Sequential()
    for conv_layer in range(1, m_params["conv_layers"] + 1):
        if conv_layer == 1:
            model.add(
                Conv1D(filters=m_params["filters"][conv_layer - 1], kernel_size=m_params["kernel_size"][conv_layer - 1],
                       strides=m_params["strides"][conv_layer - 1] , padding="same",
                       activation=m_params["activation"], input_shape=(number_of_samples, 1)))
        else:
            model.add(
                Conv1D(filters=m_params["filters"][conv_layer - 1], kernel_size=m_params["kernel_size"][conv_layer - 1],
                       strides=m_params["strides"][conv_layer - 1],  padding="same",
                       activation=m_params["activation"]))

        model.add(m_params["pooling_layers"][conv_layer - 1])
        model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(m_params["neurons"], activation=m_params["activation"], kernel_initializer='he_uniform',
                    input_shape=(number_of_samples,)))
    if dropout:
        model.add(Dropout(0.5))
    for l_i in range(m_params["dense_layers"] - 1):
        model.add(Dense(m_params["neurons"], activation=m_params["activation"], kernel_initializer='he_uniform'))
        if dropout:
            model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))
    optimizer = Adam(learning_rate=m_params["learning_rate"])
    if loss_type == "PEER_LOSS":
        model.compile(loss=peer_DMI, optimizer=optimizer, metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def run_cnn(x_profiling, x_attack, y_profiling, actual_joint_train, actual_joint_test, epochs, classes, trained_folder,
            base, model_ind, dropout, loss_type, g_or_l, dataset):
    num_samples = x_profiling.shape[1]
    x_profiling = x_profiling.reshape((x_profiling.shape[0], x_profiling.shape[1], 1))
    if g_or_l == "generate":
        cnn_params = cnn_network_parameters()
        with open(f'{base}/Model_{model_ind}.pkl', 'wb') as file:
            pickle.dump(cnn_params, file)
    elif g_or_l == "load":
        print("loading a good model!")
        with open(f'{base}/Model_{model_ind}.pkl', 'rb') as file:
            cnn_params = pickle.load(file)

    if g_or_l in ["generate", "load"]:
        mini_batch = cnn_params["mini_batch"]
        model = cnn_random(classes, num_samples, cnn_params, dropout, loss_type)
        if g_or_l == "generate":
            initial_weights = model.get_weights()
            with open(f'{base}/initial_weights_{model_ind}.pkl', 'wb') as f:
                pickle.dump(initial_weights, f)
        elif g_or_l == "load":
            with open(f'{base}/initial_weights_{model_ind}.pkl', 'rb') as f:
                initial_weights = pickle.load(f)
            model.set_weights(initial_weights)
        model.fit(
            x=x_profiling,
            y=to_categorical(y_profiling, num_classes=classes),
            batch_size=mini_batch,
            verbose=0,
            epochs=epochs,
            shuffle=True,
            callbacks=[])

    elif g_or_l == "load_trained":
        print(f'Loading model {model_ind}')
        model = load_model(trained_folder + f"trained_models/{loss_type}_{dropout}_{model_ind}.h5")

    else:
        print("Error!")
        exit(-1)

    predictions = np.array(model.predict(x_profiling))
    predicted_hw_p = np.argmax(predictions, axis=1)

    predictions = np.array(model.predict(x_attack))
    predicted_hw_t = np.argmax(predictions, axis=1)

    if dataset in ("Kyber", "Chipwhisperer"):
        profile_total_acc, profile_detailed_acc_m, profile_detailed_acc_y = \
            customized_accuracy(predicted_hw_p, actual_joint_train, classes)
        attack_total_acc, attack_detailed_acc_m, attack_detailed_acc_y = \
            customized_accuracy(predicted_hw_t, actual_joint_test, classes)
        accuracy = {
            'profile_acc_m': profile_detailed_acc_m,
            'profile_acc_y': profile_detailed_acc_y,
            'profile_acc': profile_total_acc,
            'attack_acc_m': attack_detailed_acc_m,
            'attack_acc_y': attack_detailed_acc_y,
            'attack_acc': attack_total_acc
        }
    else:
        profile_total_acc, profile_detailed_acc_m1, profile_detailed_acc_m2, profile_detailed_acc_y, profile_acc_m1, profile_acc_m2, profile_acc_y = \
            customized_accuracy_ascon(predicted_hw_p, actual_joint_train)
        attack_total_acc, attack_detailed_acc_m1, attack_detailed_acc_m2, attack_detailed_acc_y, attack_acc_m1, attack_acc_m2, attack_acc_y = \
            customized_accuracy_ascon(predicted_hw_t, actual_joint_test)
        accuracy = {
            'profile_acc_m1': profile_detailed_acc_m1,
            'profile_acc_m2': profile_detailed_acc_m2,
            'profile_acc_y': profile_detailed_acc_y,
            'profile_acc': profile_total_acc,
            'profile_acc_single_m1': profile_acc_m1,
            'profile_acc_single_m2': profile_acc_m2,
            'profile_acc_single_y': profile_acc_y,
            'attack_acc_m1': attack_detailed_acc_m1,
            'attack_acc_m2': attack_detailed_acc_m2,
            'attack_acc_y': attack_detailed_acc_y,
            'attack_acc': attack_total_acc,
            'attack_acc_single_m1': attack_acc_m1,
            'attack_acc_single_m2': attack_acc_m2,
            'attack_acc_single_y': attack_acc_y
        }

    del model
    backend.clear_session()
    gc.collect()

    return predicted_hw_t, accuracy

