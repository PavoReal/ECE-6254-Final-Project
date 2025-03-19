import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def get_all_symbols(dataset_dir):
    train_dir = os.path.join(dataset_dir, "split", "train")
    symbols = []
    
    if os.path.exists(train_dir):
        for file in os.listdir(train_dir):
            if file.endswith(".csv"):
                symbols.append(file.replace(".csv", ""))
    
    return symbols

def download_and_split_data(symbol, output_dir="./yfinance_dataset", train_ratio=0.8):
    train_dir = os.path.join(output_dir, "split", "train")
    test_dir = os.path.join(output_dir, "split", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"Downloading data for {symbol}...")
    ticker = yf.Ticker(symbol)
    
    try:
        data = ticker.history(period="max")
        
        if len(data) == 0:
            print(f"No data available for {symbol}")
            return
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Rename columns to match original dataset format
        data = data.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        data = data.sort_values('Date')
        
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        train_file = os.path.join(train_dir, f"{symbol}.csv")
        test_file = os.path.join(test_dir, f"{symbol}.csv")
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        print(f"Data saved for {symbol}")
    
    except Exception as e:
        print(f"Error downloading {symbol}: {str(e)}")
    
    print(f'Files split and saved ./yfinance_dataset/split')