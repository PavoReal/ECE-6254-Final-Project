import matplotlib.pyplot as plt
import tensorflow as tf
import glob, os
import pickle

def test_main(model_path):
    model_load_path       = model_path + '.keras'
    shared_data_load_path = model_path + '.pkl'

    # Load model
    model = tf.keras.models.load_model(model_load_path)

    print(f'Loaded model {model_load_path} from disk')

    # Load shared data
    with open(shared_data_load_path, "rb") as f:
        shared_data = pickle.load(f)

    test_seq   = shared_data["test_seq"]
    test_label = shared_data["test_label"]
    scaler     = shared_data["scaler"]

    print(f'Loaded shared data {shared_data_load_path} from disk')

    # Prediction & inverse scaling
    test_pred     = model.predict(test_seq)
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

    plt.savefig('predictions.png')

    plt.show()

if __name__ == '__main__':
    model_path = './models/test-model'

    test_main(model_path);