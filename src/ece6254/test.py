import matplotlib.pyplot as plt
import tensorflow as tf
import glob, os
import pickle
import pandas as pd

from . import dataset

def load_model_files(model_path, data_name, data_dir):
    ignored, dataset_load_path = dataset.get_dataset_files(data_name, data_dir)

    model_load_path       = model_path + '.keras'
    shared_data_load_path = model_path + '.pkl'

    # Load model
    model = tf.keras.models.load_model(model_load_path)

    print(f'Loaded model {model_load_path} from disk')

    testing_data = pd.read_csv(dataset_load_path)

    # Load shared data
    with open(shared_data_load_path, "rb") as f:
        shared_data = pickle.load(f)

    features   = shared_data["features"];
    scaler     = shared_data["scaler"]
    seq_length = shared_data["seq_length"]

    test_data  = scaler.transform(testing_data[features])

    test_seq, test_label = dataset.create_sequence(test_data, seq_length)

    print(f'Loaded shared data {shared_data_load_path} from disk')

    return model, test_data, test_seq, test_label, scaler

def test_main(model_path, data_name, data_dir):

    model, test_data, test_seq, test_label, scaler = load_model_files(model_path, data_name, data_dir)

    # Prediction & inverse scaling
    test_pred     = model.predict(test_seq)
    test_inv_pred = scaler.inverse_transform(test_pred)

    # if test_label.ndim == 1 or test_label.shape[1] != 1:
    #     test_label = test_label.reshape(-1, 1)

    test_inv_label = scaler.inverse_transform(test_label)

    # Xkcd style, cool kids only
    # plt.xkcd()

    # Plot predicted vs actual
    plt.figure(figsize=(12,6))
    plt.plot(test_inv_label, color='blue', label='Actual Close Price')
    plt.plot(test_inv_pred, color='red', label='Predicted Close Price')
    plt.title(f'Close Price Prediction {data_name}')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()

    os.makedirs('./figures', exist_ok=True)

    plt.savefig(f'./figures/predictions-{data_name}.png')

    plt.show()

def compare_main(model_path_a, model_path_b, data_name, data_dir):

    model_a, test_data_a, test_seq_a, test_label_a, scaler_a = load_model_files(model_path_a, data_name, data_dir)
    model_b, test_data_b, test_seq_b, test_label_b, scaler_b = load_model_files(model_path_b, data_name, data_dir)

    test_pred_a = model_a.predict(test_seq_a)
    test_pred_b = model_b.predict(test_seq_b)

    test_inv_pred_a = scaler_a.inverse_transform(test_pred_a)
    test_inv_pred_b = scaler_b.inverse_transform(test_pred_b)

    # Do we need to keep these?
    # if test_label_a.ndim == 1 or test_label_a.shape[1] != 1:
    #     test_label_a = test_label_a.reshape(-1, 1)

    # if test_label_b.ndim == 1 or test_label_b.shape[1] != 1:
    #     test_label_b = test_label_b.reshape(-1, 1)

    test_inv_label_a = scaler_a.inverse_transform(test_label_a)
    test_inv_label_b = scaler_b.inverse_transform(test_label_b)

    longest_path = max(len(model_path_a), len(model_path_b))

    plt.figure(figsize=(12,6))
    plt.plot(test_inv_label_a, color='blue',  label='Actual Close Price')
    plt.plot(test_inv_pred_a,  color='red',   label=f'{model_path_a.ljust(longest_path)} Predicted Close Price')
    plt.plot(test_inv_pred_b,  color='green', label=f'{model_path_b.ljust(longest_path)} Predicted Close Price')

    plt.title(f'Close Price Prediction {data_name}')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()

    name_a = os.path.basename(model_path_a)
    name_b = os.path.basename(model_path_b)

    os.makedirs('./figures', exist_ok=True)

    plt.savefig(f'./figures/compare-{name_a}-{name_b}-{data_name}.png')

    plt.show()