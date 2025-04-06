import matplotlib.pyplot as plt
import tensorflow as tf
import glob, os
import pickle
import pandas as pd
import numpy as np
from ece6254 import randomForest

from . import dataset

def load_model_files(model_path, data_name, data_dir):
    ignored, dataset_load_path = dataset.get_dataset_files(data_name, data_dir)

    model_load_path       = model_path + '.keras'
    shared_data_load_path = model_path + '.pkl'
    rf_model_path = model_path + '.pkl'

    # Load model
    try:
        if os.path.exists(model_load_path):
            model = tf.keras.models.load_model(model_load_path)
            print(f'Loaded model {model_load_path} from disk')
        elif os.path.exists(rf_model_path):
            with open(rf_model_path, 'rb') as f:
                model = pickle.load(f)
            print(f'Loaded RF pickle model {rf_model_path} from disk')
        else:
            print(f"Error: Model file not found at either {model_load_path} or {rf_model_path}")
            return None, None, None, None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None, None

    testing_data = pd.read_csv(dataset_load_path)

    # Load shared data
    with open(shared_data_load_path, "rb") as f:
        shared_data = pickle.load(f)

    features   = shared_data["features"]
    scaler     = shared_data["scaler"]
    seq_length = shared_data["seq_length"]
    lag        = shared_data["lag"]
    
    test_data  = scaler.transform(testing_data[features])

    if lag is not None and 'randForest' in model_path:
        X_test_lag, y_test_lag = randomForest.create_lag(lag, test_data)
        test_seq = X_test_lag
        test_label = testing_data[features[0]][lag:]
    elif seq_length is not None and 'randForest' not in model_path:
        test_seq, test_label = dataset.create_sequence(test_data, seq_length)
    else:
        print("Error: Inconsistent or missing sequence/lag information for the model type.")
        return None, None, None, None, None
    
    print(f'Loaded shared data {shared_data_load_path} from disk')
    return model, test_data, test_seq, test_label, scaler

def test_main(model_path, data_name, data_dir):

    model, test_data, test_seq, test_label, scaler = load_model_files(model_path, data_name, data_dir)

    # debugging print statements
    # print(f"test_seq shape: {test_seq.shape}")
    # print(f"test_label shape: {test_label.shape}")
    # print(f"test_pred shape: {model.predict(test_seq).shape}")
    # Prediction & inverse scaling
    test_pred = model.predict(test_seq)

    # put the predictions in the correct column (assuming we're predicting Close)
    with open(model_path + '.pkl', "rb") as f:
        shared_data = pickle.load(f)
    features = shared_data["features"]

    # only selecting the 'Close' feature
    target_feature_index = features.index('Close') if 'Close' in features else 0

    # if test_label.ndim == 1 or test_label.shape[1] != 1:
    #     test_label = test_label.reshape(-1, 1)

    # correctly sizing the pred and label test arrays
    dummy_pred_array = np.zeros((test_pred.shape[0], len(features)))
    dummy_pred_array[:, target_feature_index] = test_pred.flatten()
    test_inv_pred = scaler.inverse_transform(dummy_pred_array)
    test_inv_pred = test_inv_pred[:, target_feature_index].reshape(-1, 1)

    test_inv_label = scaler.inverse_transform(test_label)
    test_inv_label = test_inv_label[:, target_feature_index].reshape(-1, 1)

    # Xkcd style, cool kids only
    plt.xkcd()

    # Plot predicted vs actual
    plt.figure(figsize=(12,6))
    plt.plot(test_inv_label, color='blue', label='Actual Close Price')
    plt.plot(test_inv_pred, color='red', label='Predicted Close Price')
    plt.title(f'Close Price Prediction {data_name}')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()

    save_dir = './predictions'

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f'{save_dir}/predictions-{os.path.basename(model_path)}-{data_name}.png')

    plt.show()

def compare_main(model_paths, data_name, data_dir):
    # List to store all model predictions and their names
    predictions = []
    model_names = []
    longest_path = 0

    # Load each model and get its predictions
    for model_path in model_paths:
        model, test_data, test_seq, test_label, scaler = load_model_files(model_path, data_name, data_dir)
        test_pred = model.predict(test_seq)

        # Load features from shared data
        with open(model_path + '.pkl', "rb") as f:
            shared_data = pickle.load(f)
        features = shared_data["features"]

        # Only selecting the 'Close' feature
        target_feature_index = features.index('Close') if 'Close' in features else 0

        # Correctly sizing the pred and label test arrays
        dummy_pred_array = np.zeros((test_pred.shape[0], len(features)))
        dummy_pred_array[:, target_feature_index] = test_pred.flatten()
        test_inv_pred = scaler.inverse_transform(dummy_pred_array)
        test_inv_pred = test_inv_pred[:, target_feature_index].reshape(-1, 1)

        test_inv_label = scaler.inverse_transform(test_label)
        test_inv_label = test_inv_label[:, target_feature_index].reshape(-1, 1)
        
        predictions.append(test_inv_pred)
        model_names.append(os.path.basename(model_path))
        longest_path = max(longest_path, len(model_path))

    # Plot setup
    plt.figure(figsize=(12,6))
    
    # Plot actual values (using the last test_inv_label since they're all the same)
    plt.plot(test_inv_label, color='blue', label='Actual Close Price')
    
    # Define a color map for predictions
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        color = colors[i % len(colors)]
        plt.plot(pred, color=color, label=f'{name.ljust(longest_path)} Predicted Close Price')

    plt.title(f'Close Price Prediction {data_name}')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()

    # Create filename from all model names
    filename = f'./figures/compare-{"-".join(model_names)}-{data_name}.png'
    os.makedirs('./figures', exist_ok=True)
    plt.savefig(filename)

    plt.show()
