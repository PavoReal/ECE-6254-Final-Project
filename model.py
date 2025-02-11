import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Input, BatchNormalization
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_sequence(dataset, seq_length=10):
    if len(dataset) <= seq_length:
        print(f"Dataset length {len(dataset)} is too short for sequence length {seq_length}.")
        return np.empty((0, seq_length, dataset.shape[1])), np.empty((0, dataset.shape[1]))
    
    seqs, labels = [], []

    for i in range(seq_length, len(dataset)):
        seqs.append(dataset[i - seq_length:i, :])
        labels.append(dataset[i, :])

    return np.array(seqs), np.array(labels)

# Load dataset
data = pd.read_csv('./raw/2/stocks/INTC.csv')

# Use only one feature for now
features = ['Close']
scaler = MinMaxScaler()
stock_data = scaler.fit_transform(data[features])

if len(stock_data) < 2:
    print("Not enough data to create training sequences.")
    sys.exit(1)

# Train/test split
train_size = int(0.8 * len(stock_data))
train_data = stock_data[:train_size]
test_data  = stock_data[train_size:]

# Set sequence length
default_seq_length = 50
if len(train_data) <= default_seq_length:
    default_seq_length = max(1, len(train_data) - 1)

train_seq, train_label = create_sequence(train_data, default_seq_length)
test_seq, test_label   = create_sequence(test_data, default_seq_length)

if train_seq.size == 0 or test_seq.size == 0:
    print("Not enough data for sequence creation.")
    sys.exit(1)

train_seq = np.array(train_seq, dtype=np.float32)
test_seq = np.array(test_seq, dtype=np.float32)

model = Sequential()
model.add(Input(shape=(default_seq_length, train_data.shape[1])))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

model.fit(train_seq, train_label, epochs=80, validation_data=(test_seq, test_label),
          callbacks=[early_stop, reduce_lr], verbose=1)

# Prediction & inverse scaling
test_pred = model.predict(test_seq)
test_inv_pred = scaler.inverse_transform(test_pred)

if test_label.ndim == 1 or test_label.shape[1] != 1:
    test_label = test_label.reshape(-1, 1)

test_inv_label = scaler.inverse_transform(test_label)

# Plot predicted vs actual
plt.figure(figsize=(12,6))
plt.plot(test_inv_label, color='blue', label='Actual Close Price')
plt.plot(test_inv_pred, color='red', label='Predicted Close Price')
plt.title('Close Price Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()
