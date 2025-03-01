import os
import kagglehub
import shutil
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def print_dataset_list(folder):
    always_exclude = ['symbols_valid_meta.csv']

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file in always_exclude:
                continue;

            print(f'{file}')

def get_training_dataset_path(ticker, data_dir):
    return os.path.join(data_dir, 'split', 'train', ticker + '.csv')

def get_testing_dataset_path(ticker, data_dir):
    return os.path.join(data_dir, 'split', 'test', ticker + '.csv')

def get_dataset_files(ticker, data_dir):
    return get_training_dataset_path(ticker, data_dir), get_testing_dataset_path(ticker, data_dir)

def create_sequence(dataset, seq_length):
    if len(dataset) <= seq_length:
        print(f"Dataset length {len(dataset)} is too short for sequence length {seq_length}.")
        return np.empty((0, seq_length, dataset.shape[1])), np.empty((0, dataset.shape[1]))
    
    seqs, labels = [], []

    for i in range(seq_length, len(dataset)):
        seqs.append(dataset[i - seq_length:i, :])
        labels.append(dataset[i, :])

    return np.array(seqs), np.array(labels)

def split_data_csv(pre_path, output_dir, name, train_ratio=0.8):
    # Load the dataset
    raw_file = pd.read_csv(pre_path, parse_dates=['Date'])

    file_sorted = raw_file.sort_values('Date')

    split_idx = int(len(file_sorted) * 0.8)

    train_set = file_sorted.iloc[:split_idx]
    test_set  = file_sorted.iloc[split_idx:]

    train_set.to_csv(os.path.join(output_dir, 'train', name + '.csv'), index=False)
    test_set.to_csv(os.path.join(output_dir,  'test',  name + '.csv'), index=False)

def download_and_save(download_path):
    raw_folder = os.path.join(download_path, 'raw')
    os.makedirs(raw_folder, exist_ok=True)
    
    # Download dataset
    downloaded_path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
    
    print('Download done')

    # Copy the downloaded file to raw directory
    raw_file_path = os.path.join(raw_folder, os.path.basename(downloaded_path))

    if os.path.isdir(downloaded_path):
        # If target directory already exists, remove it to avoid copytree error
        if os.path.exists(raw_file_path):
            shutil.rmtree(raw_file_path)

        shutil.copytree(downloaded_path, raw_file_path)
    else:
        shutil.copy(downloaded_path, raw_file_path)

    print('Extract and copy done')
    print('Starting split, this make take a while...')

    ticker_folders    = ['stocks', 'etfs']
    split_output_path = os.path.join(download_path, 'split')

    os.makedirs(os.path.join(split_output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(split_output_path, 'test'),  exist_ok=True)

    os.makedirs(split_output_path, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        for folder in ticker_folders:
            folder_path = os.path.join(raw_file_path, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".csv"):
                    executor.submit(split_data_csv, os.path.join(folder_path, file), split_output_path, file.split('.')[0])

    print(f'Files split and saved to {download_path}/split')
