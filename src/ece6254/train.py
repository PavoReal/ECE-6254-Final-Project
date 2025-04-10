import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys

from datetime                   import datetime
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers    import Dense, Dropout, LSTM, Bidirectional, Input, BatchNormalization
from tensorflow.keras.models    import Sequential, load_model
from sklearn.preprocessing      import MinMaxScaler
from ece6254 import randomForest
import keras_tuner as kt
import tensorflow as tf

from . import dataset
from . import models

def get_model_arch(name):
    for arch in models.model_arch:
        if arch["name"] == name:
            return arch

    raise ValueError(f'Unknown model {name}')


build_hp_model_func = None;
build_hp_model_seq_length = 20;
build_hp_model_data_shape = 1;

def build_hp_model(hp):
    assert build_hp_model_func != None;
    return build_hp_model_func(hp, build_hp_model_seq_length, build_hp_model_data_shape);

def train_main(model_file_path, data_name, data_dir, features, seq_length, epochs, model_arch, lag, tune_epocs):
    train_file_path, test_file_path = dataset.get_dataset_files(data_name, data_dir)

    # Load dataset
    training_data = pd.read_csv(train_file_path)
    testing_data  = pd.read_csv(test_file_path)

    # Fit our scaler
    scaler     = MinMaxScaler()
    train_data = scaler.fit_transform(training_data[features])
    test_data  = scaler.transform(testing_data[features])

    if len(train_data) < seq_length and "randForest" != model_arch["name"]:
        raise ValueError("Not enough input data")

    # Generate sequences

    if "randForest" != model_arch["name"]:
        train_seq, train_label = dataset.create_sequence(train_data, seq_length)
        test_seq,  test_label  = dataset.create_sequence(test_data,  seq_length)

        if train_seq.size == 0:
            raise ValueError("Training data is empty")

        train_seq = np.array(train_seq, dtype=np.float32)
        test_seq  = np.array(test_seq,  dtype=np.float32)

        model_load_path = model_file_path + '.keras'
    else:
        model_load_path = model_file_path + '.pkl'

    # If we can load the model, do that, otherwise create a new one
    if os.path.exists(model_load_path) and "randForest" != model_arch["name"]:
        model = load_model(model_load_path)
        print('Loaded Keras model from disk')
    else:
        if "randForest" == model_arch["name"]:
            model = model_arch["func"](train_data, lag)
        elif model_arch["tune"] == True:
            global build_hp_model_func;
            build_hp_model_func       = model_arch["func"];

            global build_hp_model_data_shape;
            build_hp_model_data_shape = train_data.shape[1];

            global build_hp_model_seq_length;
            build_hp_model_seq_length = seq_length;

            tuner = kt.RandomSearch(
                build_hp_model,
                objective='val_mse',
                max_trials=tune_epocs,
                executions_per_trial=2,
                directory='tuner_work',
                project_name=model_arch["name"]
            )

            #tuner = kt.Hyperband(
            #    build_hp_model,
            #    objective='val_mse',
            #    factor=3,
            #    executions_per_trial=2,
            #    directory='tuner_work',
            #    project_name=model_arch["name"]
            #)

            tuner.search(train_seq, train_label, epochs=20, validation_data=(test_seq, test_label));

            model = tuner.get_best_models(num_models=1)[0];
        else:
            model = model_arch["func"](seq_length, train_data.shape[1])

        print('Compiled model')

    if model_arch["name"] != "randForest":
        model.summary()
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        model.fit(train_seq, train_label, epochs=epochs, batch_size=256, validation_data=(test_seq, test_label),
              callbacks=[early_stop, reduce_lr], verbose=1)

        shated_data_path = model_file_path + '.pkl'

        # Save model and shared data
        model.save(model_load_path)

        print('Model saved to disk')
    else:
        rd_save_path = model_file_path + '.pkl';
        with open(rd_save_path, 'wb') as f:
            pickle.dump(model, f);

            print("Random forest model saved");
        
        shated_data_path = model_file_path + '.dat.pkl'

    shared_data = {'scaler': scaler, 'seq_length': seq_length, 'features': features, 'lag': lag}

    with open(shated_data_path, 'wb') as f:
        pickle.dump(shared_data, f)

    print("Shared data saved to disk")

