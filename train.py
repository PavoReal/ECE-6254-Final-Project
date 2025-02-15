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

def create_sequence(dataset, seq_length):
    if len(dataset) <= seq_length:
        print(f"Dataset length {len(dataset)} is too short for sequence length {seq_length}.")
        return np.empty((0, seq_length, dataset.shape[1])), np.empty((0, dataset.shape[1]))
    
    seqs, labels = [], []

    for i in range(seq_length, len(dataset)):
        seqs.append(dataset[i - seq_length:i, :])
        labels.append(dataset[i, :])

    return np.array(seqs), np.array(labels)

def create_model(seq_length, data_shape):
    model = Sequential()
    model.add(Input(shape=(seq_length, data_shape)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(1))

    return model

def train_main(model_file_path, data_file_path, features=['Close'], seq_length=30, epochs=80):
    # Load dataset
    data = pd.read_csv(data_file_path)

    # Fit our scaler
    scaler     = MinMaxScaler()
    stock_data = scaler.fit_transform(data[features])

    if len(stock_data) < seq_length:
        raise ValueError("Not enough input data")

    train_test_ratio = 0.8;

    # Train/test split
    train_size = int(train_test_ratio * len(stock_data))
    train_data = stock_data[:train_size]
    test_data  = stock_data[train_size:]

    # Generate sequences
    train_seq, train_label = create_sequence(train_data, seq_length)
    test_seq,  test_label  = create_sequence(test_data, seq_length)

    if train_seq.size == 0:
        raise ValueError("Training data is empty")

    train_seq = np.array(train_seq, dtype=np.float32)
    test_seq  = np.array(test_seq,  dtype=np.float32)

    model_load_path = model_file_path + '.keras';

    # If we can load the model, do that, otherwise create a new one
    if os.path.exists(model_load_path):
        model = load_model(model_load_path);
        print('Loaded model from disk')
    else:
        model = create_model(seq_length, train_data.shape[1])

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
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
    shared_data      = {'test_seq': test_seq, 'test_label': test_label, 'scaler': scaler, 'seq_length': seq_length}

    with open(shated_data_path, 'wb') as f:
        pickle.dump(shared_data, f)

    print("Shared data saved to disk")

if __name__ == '__main__':
    data_file_path = f'./raw/2/stocks/INTC.csv';
    start_time     = datetime.now().strftime('%Y:%m:%d')
    # file_name      = f'./models/test-model-{start_time}'
    file_name = f'./models/test-model'

    os.makedirs('./models/', exist_ok=True)

    train_main(file_name, data_file_path);