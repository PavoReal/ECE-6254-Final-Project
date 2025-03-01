from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import Dense, Dropout, LSTM, Bidirectional, Input, BatchNormalization

def create_model_gp(seq_length, data_shape):
    model = Sequential()
    model.add(Input(shape=(seq_length, data_shape)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    return model

def create_model_anu(seq_length, data_shape):
    model = Sequential()
    model.add(Input(shape=(seq_length, data_shape)))
    model.add(LSTM(units = 50, activation = 'relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 60, activation = 'relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units = 80, activation = 'relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units = 120, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    return model

def create_ann_model(seq_length, data_shape):
    """
    Creates the ANN model from Table 1 of the paper.
    - Architecture: Two dense layers with 10 neurons each, dropout 0.5, output dense with 1 neuron
    """
    model = Sequential([
        Input(shape=(seq_length, data_shape)),
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_lstm1d_model(seq_length, data_shape):
    """
    Creates the LSTM1D model from Table 2 of the paper.
    - Architecture: One LSTM layer with 10 units, dense with 10 neurons, dropout 0.5, output dense with 1 neuron
    """
    model = Sequential([
        Input(shape=(seq_length, data_shape)),
        LSTM(10),
        Dense(10, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_lstm2d_model(seq_length, data_shape):
    """
    Creates the LSTM2D model from Table 3 of the paper.
    - Architecture: Two LSTM layers with 10 units each, dense with 10 neurons, dropout 0.5, output dense with 1 neuron
    """
    model = Sequential([
        Input(shape=(seq_length, data_shape)),
        LSTM(10, return_sequences=True),
        LSTM(10),
        Dense(10, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_lstm3d_model(seq_length, data_shape):
    """
    Creates the LSTM3D model from Table 4 of the paper.
    - Architecture: Three LSTM layers with 10 units each, dense with 10 neurons, dropout 0.5, output dense with 1 neuron
    """
    model = Sequential([
        Input(shape=(seq_length, data_shape)),
        LSTM(10, return_sequences=True),
        LSTM(10, return_sequences=True),
        LSTM(10),
        Dense(10, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# LSTM-GA and LSTM-ARO Model
def create_lstm_optimized_model(seq_length, data_shape, x0, x1, x2, x3, x4, x5, x6, optimizer, learning_rate):
    """
    Creates the LSTM-GA or LSTM-ARO model from Table 5 of the paper.
    - Architecture: Variable number of LSTM layers (1 to 3), dense, dropout, output dense with 1 neuron
    - Parameters:
        x0: Number of units in the first LSTM layer (1 to 20)
        x1: Binary (0 or 1), whether the second LSTM layer exists
        x2: Number of units in the second LSTM layer if x1=1 (1 to 20)
        x3: Binary (0 or 1), whether the third LSTM layer exists (requires x1=1)
        x4: Number of units in the third LSTM layer if x3=1 (1 to 20)
        x5: Number of units in the dense layer (1 to 20)
        x6: Dropout rate (e.g., 0.3, 0.4, 0.5, 0.6, 0.7)
        optimizer: String, e.g., 'adam', 'sgd', 'rmsprop'
        learning_rate: Float, e.g., 0.1, 0.01, 0.001, 0.0001, 0.00001
    """
    model = Sequential()
    model.add(Input(shape=(seq_length, data_shape)))

    if x1 == 0:
        # Only one LSTM layer
        model.add(LSTM(x0, return_sequences=False))
    else:
        # First LSTM layer, followed by more if specified
        model.add(LSTM(x0, return_sequences=True))
        if x3 == 0:
            # Two LSTM layers
            model.add(LSTM(x2, return_sequences=False))
        else:
            # Three LSTM layers
            model.add(LSTM(x2, return_sequences=True))
            model.add(LSTM(x4, return_sequences=False))

    model.add(Dense(x5, activation='relu'))
    model.add(Dropout(x6))
    model.add(Dense(1, activation='linear'))

    # Configure the optimizer
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer: choose 'adam', 'sgd', or 'rmsprop'")

    model.compile(optimizer=opt, loss='mse')
    return model

model_arch = [
    {'name': 'lstm1d', 'desc': 'Creates the LSTM1D model from Table 2 of the paper', 'func': create_lstm1d_model},
    {'name': 'lstm2d', 'desc': 'Creates the LSTM2D model from Table 3 of the paper', 'func': create_lstm2d_model},
    {'name': 'lstm3d', 'desc': 'Creates the LSTM3D model from Table 4 of the paper', 'func': create_lstm3d_model},
    {'name': 'ann', 'desc': 'Creates the ANN model from Table 1 of the paper', 'func': create_ann_model},
    {'name': 'anu', 'desc': 'Initial test model from anush-lstm branch', 'func': create_model_anu},
    {'name': 'gp', 'desc': 'Initial test model from gp-lstm-test branch', 'func': create_model_gp}
]

def print_arch_list():
    longest_name = 0
    longet_desc  = 0

    for arch in models.model_arch:
        longest_name = max(longest_name, len(arch["name"]))
        longet_desc  = max(longet_desc, len(arch["desc"]))

    longest_name = max(14, longest_name)

    total_width = longest_name + longet_desc + 4

    print("~" * total_width)

    print(f'{"name".ljust(longest_name)} -- {"description".ljust(longet_desc)}')
    print(f'{"-" * (longest_name + longet_desc + 4)}')

    for arch in models.model_arch:
        
        default = "";
        if arch == models.model_arch[0]:
            default = " (default)"

        print(f'{(arch["name"] + default).ljust(longest_name)} -- {arch["desc"].ljust(longet_desc)}')

    print("~" * total_width)
