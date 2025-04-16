import matplotlib.pyplot as plt
import tensorflow as tf
import os
import glob
import pickle
import pandas as pd
import numpy as np
import csv
from ece6254 import randomForest
from sklearn.metrics import mean_squared_error, mean_absolute_error

from . import dataset
from . import randomForest

def load_model_files(model_path):
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
            return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    # Load shared data
    with open(shared_data_load_path, "rb") as f:
        shared_data = pickle.load(f)
    print(f'Loaded shared data {shared_data_load_path} from disk')

    return model, shared_data;

def load_data_files(data_name, data_dir, shared_data, is_random_forest):
    _, dataset_load_path = dataset.get_dataset_files(data_name, data_dir)

    testing_data = pd.read_csv(dataset_load_path)

    features   = shared_data["features"]
    scaler     = shared_data["scaler"]
    seq_length = shared_data["seq_length"]
    lag        = shared_data["lag"]
    
    test_data  = scaler.transform(testing_data[features])

    test_dates_full = pd.to_datetime(testing_data['Date']).values;
    test_close_full = testing_data['Close'].values;

    if lag is not None and is_random_forest:
        X_test_lag, y_test_lag = randomForest.create_lag(lag, test_data)
        test_seq   = X_test_lag
        test_label = testing_data[features[0]][lag:].values.reshape(-1, 1)
        test_dates = pd.to_datetime(testing_data['Date'][lag:]).values
    elif seq_length is not None and not is_random_forest:
        test_seq, test_label = dataset.create_sequence(test_data, seq_length)
        test_dates = pd.to_datetime(testing_data['Date'][seq_length:]).values
    else:
        raise ValueError("Error: Inconsistent or missing sequence/lag information for the model type.")
    
    return test_data, test_seq, test_label, scaler, test_dates, test_dates_full, test_close_full

def summarize_and_plot_model_performance(base_dir='./compare'):
    csv_files = glob.glob(os.path.join(base_dir, '*', '*', 'modelEvalStats.csv'))
    all_metrics = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                all_metrics.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if all_metrics:
        metrics_df = pd.concat(all_metrics)

        error_stats = metrics_df.groupby('Model Name')['Accuracy'].agg(['mean', 'std']).reset_index()
        error_stats = error_stats.sort_values(by='mean', ascending=False)

        model_names = error_stats['Model Name']
        means       = error_stats['mean'];
        std_devs    = error_stats['std'];

        plt.figure(figsize=(12, 6))
        plt.bar(model_names, means, yerr=std_devs, capsize=5, color='skyblue', edgecolor='black', width=0.4)
        plt.title('Average Accuracy by Model')
        plt.xlabel('Model')
        plt.ylabel('Average Accuracy (%)')
        plt.xticks(rotation=45, ha='right')

        for i, (mean, std) in enumerate(zip(means, std_devs)):
            plt.text(i + 0.22, mean + 0.2, f'{mean:.4f}', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'avg_accuracy.svg'))
        plt.show()
        plt.close()
    else:
        print("No metrics data found.")

def test_main(model_path, data_name, data_dir):
    model, shared_data = load_model_files(model_path);
    test_data, test_seq, test_label, scaler, test_dates, _, _ = load_data_files(data_name, data_dir, shared_data, ("randForest" in model_path));

    test_pred = model.predict(test_seq)

    features = shared_data["features"]

    # only selecting the 'Close' feature
    target_feature_index = features.index('Close') if 'Close' in features else 0

    # correctly sizing the pred and label test arrays
    dummy_pred_array = np.zeros((test_pred.shape[0], len(features)))
    dummy_pred_array[:, target_feature_index] = test_pred.flatten()

    test_inv_pred = scaler.inverse_transform(dummy_pred_array)
    test_inv_pred = test_inv_pred[:, target_feature_index].reshape(-1, 1)

    # correctly sizing the pred and label test arrays
    dummy_pred_array = np.zeros((test_pred.shape[0], len(features)))
    dummy_pred_array[:, target_feature_index] = test_pred.flatten()
    test_inv_pred = scaler.inverse_transform(dummy_pred_array)
    test_inv_pred = test_inv_pred[:, target_feature_index].reshape(-1, 1)

    test_inv_label = scaler.inverse_transform(test_label)
    test_inv_label = test_inv_label[:, target_feature_index].reshape(-1, 1)

    # Plot predicted vs actual
    plt.figure(figsize=(12,6))
    plt.plot(test_dates, test_inv_label, color='blue', label='Actual Close Price')
    plt.plot(test_dates, test_inv_pred, color='red', label='Predicted Close Price')
    plt.title(f'Close Price Prediction {data_name}')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()

    save_dir = './predictions'

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f'{save_dir}/predictions-{os.path.basename(model_path)}-{data_name}.png')
    plt.show()

def compare_main(model_paths, data_name, data_dir):
    # Load each model and get its predictions
    for model_path in model_paths:
        model, shared_data = load_model_files(model_path);

        model_basename = os.path.basename(model_path);

        model_is_random_forest = ("randForest" in model_path);

        # List to store all model predictions and their names
        predictions  = []
        model_names  = []
        mseVec       = []
        maeVec       = []
        rmseVec      = []
        accuVec      = []
        longest_path = 0

        for ticker in data_name:
            predictions  = []
            model_names  = []
            mseVec       = []
            maeVec       = []
            rmseVec      = []
            accuVec      = []
            longest_path = 0

            test_data, test_seq, test_label, scaler, test_dates, test_dates_full, test_close_full = load_data_files(ticker, data_dir, shared_data, model_is_random_forest);

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
            
            predictions.append((test_dates, test_inv_pred))

            model_names.append(os.path.basename(model_path))
            longest_path = max(longest_path, len(model_path))

            mse, mae, rmse = model_evaluation(dummy_pred_array, test_label)
            accuracy = model_accuracy(test_inv_pred, test_inv_label)

            mseVec.append(mse)
            maeVec.append(mae)
            rmseVec.append(rmse)
            accuVec.append(accuracy)

            base_save_dir = f'./compare/{model_basename}/{ticker}/';
            os.makedirs(base_save_dir, exist_ok=True);

            # plotting model evaluation stats
            plot_model_evaluation(model_names, mseVec, maeVec, rmseVec, base_save_dir)

            # plot model accuracy
            plot_model_accuracy(model_names, accuVec, base_save_dir)

            # Plot setup
            plt.figure(figsize=(6,6))
            
            # Plot actual values (using the last test_inv_label since they're all the same)
            plt.plot(test_dates_full, test_close_full, color='blue', label='Actual Close Price')
            
            # Define a color map for predictions
            colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            for i, ((dates, pred), name) in enumerate(zip(predictions, model_names)):
                color = colors[i % len(colors)];

                plt.plot(dates, pred, color=color, label=f'{name.ljust(longest_path)}', linewidth=0.75)

            plt.title(f'{ticker} Close Price')
            plt.xlabel('')
            plt.ylabel('Close Price')
            plt.legend()

            # Create filename from all model names
            filename = f'time_series.svg'
            plt.savefig(base_save_dir + filename)
            plt.close();

            #plt.show()

            # Save metrics to CSV
            metrics_file = f'modelEvalStats.csv'
            with open(base_save_dir + metrics_file, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Model Name', 'MAE', 'MSE', 'RMSE', 'Accuracy'])
                for name, mse, mae, rmse, accuracy in zip(model_names, mseVec, maeVec, rmseVec, accuVec):
                    writer.writerow([name, mae, mse, rmse, accuracy])

    summarize_and_plot_model_performance();

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

    return meanSqErr, meanAbsErr, rmse

def get_model_complexity_order():
    """Returns the order of models from least to most complex"""
    return {
        'randforest': 0,
        'lstm1d': 1,
        'ann': 2,
        'lstm2d': 3,
        'lstm3d': 4,
        'lstm-garo': 5,
        'lstm-garo-large': 6
    }

def sort_models_by_complexity(model_names, metrics):
    """Sort models and their corresponding metrics by complexity"""
    complexity_order = get_model_complexity_order()
    
    # Create list of tuples (model_name, metric, complexity)
    model_metric_pairs = []
    for name, metric in zip(model_names, metrics):
        # Extract base model name (remove any prefixes or suffixes)
        base_name = name.lower().split('/')[-1].split('.')[0]
        complexity = complexity_order.get(base_name, 999)  # Default high complexity if not found
        model_metric_pairs.append((name, metric, complexity))
    
    # Sort by complexity
    sorted_pairs = sorted(model_metric_pairs, key=lambda x: x[2])
    
    # Unpack sorted results
    sorted_names = [pair[0] for pair in sorted_pairs]
    sorted_metrics = [pair[1] for pair in sorted_pairs]
    
    return sorted_names, sorted_metrics

def plot_model_evaluation(modelNames, mseVec, maeVec, rmseVec, base_save_dir):
    # Sort models and metrics by complexity
    modelNames, mseVec = sort_models_by_complexity(modelNames, mseVec)
    modelNames, maeVec = sort_models_by_complexity(modelNames, maeVec)
    modelNames, rmseVec = sort_models_by_complexity(modelNames, rmseVec)
    
    n_models = len(modelNames)
    x = np.arange(n_models)
    width = 0.25  # Reduced width for better spacing

    fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure size
    rects1 = ax.bar(x - width, maeVec, width, label='MAE')
    rects2 = ax.bar(x, mseVec, width, label='MSE')
    rects3 = ax.bar(x + width, rmseVec, width, label='RMSE')

    # Add labels and title
    ax.set_ylabel('Error Value')
    ax.set_title('Comparison of Model Performance Metrics (Ordered by Complexity)')
    ax.set_xticks(x)
    ax.set_xticklabels(modelNames, rotation=45, ha='right')  # Rotate labels
    ax.legend()

    # Function to add labels with better formatting
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            # Format numbers to be more concise
            if height >= 1000:
                label = f'{height/1000:.1f}k'
            elif height >= 1:
                label = f'{height:.1f}'
            else:
                label = f'{height:.3f}'
            ax.annotate(label,
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.set_yscale('log')
    plt.tight_layout()  # Adjust layout to prevent label cutoff

    # Save the figure
    filename = f'modelEvalStats.svg'
    plt.savefig(base_save_dir + filename, bbox_inches='tight')
    plt.close();
    #plt.show()

def model_accuracy(prediction_inv, test_inv):
    accuracy = 0
    datapts_total = prediction_inv.shape[0]

    for i in range(1, datapts_total):
        if ((prediction_inv[i] - prediction_inv[i - 1])* (test_inv[i] - test_inv[i - 1])) > 0:
            accuracy += 1

    accuracy_perc = (accuracy/datapts_total)*100
    return accuracy_perc

def plot_model_accuracy(model_names, accuracy, base_save_dir):
    # Sort models and accuracy by complexity
    model_names, accuracy = sort_models_by_complexity(model_names, accuracy)
    
    n_models = len(model_names)
    x = np.arange(n_models)
    width = 0.6  # Wider bars for better visibility

    fig, ax = plt.subplots(figsize=(10, 6))
    rects = ax.bar(x, accuracy, width, label='Accuracy')

    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Models')
    ax.set_title('Comparison of Model Accuracy (Ordered by Complexity)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')  # Rotate labels

    # Add labels with better formatting
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')

    autolabel(rects)

    plt.tight_layout()  # Adjust layout to prevent label cutoff

    # Save the figure
    filename = f'modelAccuracy.svg'
    plt.savefig(base_save_dir + filename, bbox_inches='tight')
    plt.close();
    #plt.show()
