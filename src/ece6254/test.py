import matplotlib.pyplot as plt
import tensorflow as tf
import glob, os
import pickle
import pandas as pd
import numpy as np
from ece6254 import randomForest
from sklearn.metrics import mean_squared_error, mean_absolute_error

from . import dataset

def load_model_files(model_path, data_name, data_dir):
    ignored, dataset_load_path = dataset.get_dataset_files(data_name, data_dir)

    model_load_path       = model_path + '.keras'
    shared_data_load_path = model_path;
    rf_model_path         = model_path + '.pkl'

    # Load model
    try:
        if os.path.exists(model_load_path):
            model = tf.keras.models.load_model(model_load_path)
            print(f'Loaded model {model_load_path} from disk')

            shared_data_load_path = shared_data_load_path + '.pkl';
        elif os.path.exists(rf_model_path):
            with open(rf_model_path, 'rb') as f:
                model = pickle.load(f)

            shared_data_load_path = shared_data_load_path + '.dat.pkl';
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
        test_seq   = X_test_lag
        test_label = testing_data[features[0]][lag:].values.reshape(-1, 1)
    elif seq_length is not None and 'randForest' not in model_path:
        test_seq, test_label = dataset.create_sequence(test_data, seq_length)
    else:
        raise ValueError("Error: Inconsistent or missing sequence/lag information for the model type.")
    
    print(f'Loaded shared data {shared_data_load_path} from disk')
    return model, test_data, test_seq, test_label, scaler, shared_data

def test_main(model_path, data_name, data_dir):
    model, test_data, test_seq, test_label, scaler, shared_data = load_model_files(model_path, data_name, data_dir)

    test_pred = model.predict(test_seq)

    features = shared_data["features"]

    # only selecting the 'Close' feature
    target_feature_index = features.index('Close') if 'Close' in features else 0

    # correctly sizing the pred and label test arrays
    dummy_pred_array = np.zeros((test_pred.shape[0], len(features)))
    dummy_pred_array[:, target_feature_index] = test_pred.flatten()

    test_inv_pred = scaler.inverse_transform(dummy_pred_array)
    test_inv_pred = test_inv_pred[:, target_feature_index].reshape(-1, 1)

    test_inv_label = scaler.inverse_transform(test_label)
    test_inv_label = test_inv_label[:, target_feature_index].reshape(-1, 1)

    # Xkcd style, cool kids only
    #plt.xkcd()

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
    predictions  = []
    model_names  = []
    mseVec = []
    maeVec = []
    rmseVec = []
    accuVec = []
    longest_path = 0

    # Load each model and get its predictions
    for model_path in model_paths:
        model, test_data, test_seq, test_label, scaler, shared_data = load_model_files(model_path, data_name, data_dir)
        
        test_pred = model.predict(test_seq)

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

        mse, mae, rmse, acc = model_evaluation(dummy_pred_array, test_label)
        mseVec.append(mse)
        maeVec.append(mae)
        rmseVec.append(rmse)
        accuVec.append(acc)

    # plotting model evaluation stats
    plot_model_evaluation(model_names, mseVec, maeVec, rmseVec)

    # Plot setup
    plt.figure(figsize=(12,6))
    
    # Plot actual values (using the last test_inv_label since they're all the same)
    plt.plot(test_inv_label, color='blue', label='Actual Close Price')
    
    # Define a color map for predictions
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (pred, name) in enumerate(zip(predictions, model_names)):
        color = colors[i % len(colors)]
        plt.plot(pred, color=color, label=f'{name.ljust(longest_path)}', linewidth=0.75)

    plt.title(f'{data_name} Close Price')
    plt.xlabel('')
    plt.ylabel('Close Price')
    plt.legend()

    # Create filename from all model names
    filename = f'./figures/compare-{"-".join(model_names)}-{data_name}.svg'
    os.makedirs('./figures', exist_ok=True)
    plt.savefig(filename)

    plt.show()

def model_evaluation(prediction, test):
    accuracy = 0
    datapts_total = prediction.shape[0]

    for i in range(1, datapts_total):
        if ((prediction[i] - prediction[i - 1])* (test[i] - test[i - 1])) > 0:
            accuracy += 1

    accuracy_perc = (accuracy/datapts_total)*100

    meanSqErr = mean_squared_error(test, prediction)
    meanAbsErr = mean_absolute_error(test, prediction)
    rmse = np.sqrt(meanSqErr)

    return meanSqErr, meanAbsErr, rmse, accuracy_perc

def plot_model_evaluation(modelNames, mseVec, maeVec, rmseVec):
    n_models = len(modelNames)
    x = np.arange(n_models)
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - 1.5*width, maeVec, width, label='MAE')
    rects2 = ax.bar(x - 0.5*width, mseVec, width, label='MSE')
    rects3 = ax.bar(x + 0.5*width, rmseVec, width, label='RMSE')
    # rects4 = ax.bar(x + 1.5*width, accuracyPerc, width, label='Accuracy (%)')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Error/Accuracy Value')
    ax.set_title('Comparison of Model Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(modelNames)
    ax.legend()

    ax.bar_label(rects1, fmt='%.2f', padding=3)
    ax.bar_label(rects2, fmt='%.2f', padding=3)
    ax.bar_label(rects3, fmt='%.2f', padding=3)
    # ax.bar_label(rects4, fmt='%.2f', padding=3)

    fig.tight_layout()
    plt.figure(figsize=(12,6))

    # Create filename from all model names
    filename = f'modelEvalStats.png'
    os.makedirs('./figures', exist_ok=True)
    plt.savefig(filename)
    plt.show()
