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

from . import dataset
from . import models

def get_model_arch(name):
    for arch in models.model_arch:
        if arch["name"] == name:
            return arch

    raise ValueError(f'Unknown model {name}')

    return model

def train_main(model_file_path, data_name, data_dir, features, seq_length, epochs, model_arch, lag):
    train_file_path, test_file_path = dataset.get_dataset_files(data_name, data_dir)

    # Load dataset
    training_data = pd.read_csv(train_file_path)
    testing_data  = pd.read_csv(test_file_path)

    # Fit our scaler
    train_scaler     = MinMaxScaler()
    train_data = train_scaler.fit_transform(training_data[features])

    test_scaler = MinMaxScaler()
    test_data  = test_scaler.fit_transform(testing_data[features])

    if len(train_data) < seq_length:
        raise ValueError("Not enough input data")

    # Generate sequences
    train_seq, train_label = dataset.create_sequence(train_data, seq_length)
    test_seq,  test_label  = dataset.create_sequence(test_data,  seq_length)

    if train_seq.size == 0:
        raise ValueError("Training data is empty")

    train_seq = np.array(train_seq, dtype=np.float32)
    test_seq  = np.array(test_seq,  dtype=np.float32)

    model_load_path = model_file_path + '.keras'

    # If we can load the model, do that, otherwise create a new one
    if os.path.exists(model_load_path):
        model = load_model(model_load_path)
        print('Loaded model from disk')
    else:
        if model_arch["name"] == "randForest":
            model = model_arch["func"](train_data, lag, features)
        
        elif model_arch["name"] == "lstmGAandARO":
            model = model_arch["func"](seq_length, train_data.shape[1])
        
        else:
            model = model_arch["func"](seq_length, train_data.shape[1])
        print('Compiled new model')


    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    model.fit(train_seq, train_label, epochs=epochs, validation_data=(test_seq, test_label),
              callbacks=[early_stop, reduce_lr], verbose=1)

    # Save model and shared data
    model.save(model_load_path)

    print('Model saved to disk')

    shated_data_path = model_file_path + '.pkl'
    shared_data      = {'scaler': train_scaler, 'seq_length': seq_length, 'features': features}

    with open(shated_data_path, 'wb') as f:
        pickle.dump(shared_data, f)

    print("Shared data saved to disk")
