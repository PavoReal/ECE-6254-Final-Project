import matplotlib.pyplot as plt
import tensorflow as tf
import glob, os
import pickle
import pandas as pd

from . import dataset

def test_main(model_path, data_name, data_dir):

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

    # Prediction & inverse scaling
    test_pred     = model.predict(test_seq)
    test_inv_pred = scaler.inverse_transform(test_pred)

    if test_label.ndim == 1 or test_label.shape[1] != 1:
        test_label = test_label.reshape(-1, 1)

    test_inv_label = scaler.inverse_transform(test_label)

    basename = os.path.basename(model_load_path).replace(".keras", "")

    # Plot predicted vs actual
    plt.figure(figsize=(12,6))
    plt.plot(test_inv_label, color='blue', label='Actual Close Price')
    plt.plot(test_inv_pred, color='red', label='Predicted Close Price')
    plt.title(f'Close Price Prediction {basename}')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()

    plt.savefig(f'predictions-{basename}.png')

    plt.show()
