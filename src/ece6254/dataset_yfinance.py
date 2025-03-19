import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def get_all_symbols(dataset_dir):
    """
    Get all symbols from the current dataset directory.
    
    Args:
        dataset_dir (str): Path to the dataset directory
    
    Returns:
        list: List of stock symbols
    """
    train_dir = os.path.join(dataset_dir, "split", "train")
    symbols = []
    
    if os.path.exists(train_dir):
        for file in os.listdir(train_dir):
            if file.endswith(".csv"):
                symbols.append(file.replace(".csv", ""))
    
    return symbols

def download_and_split_data(symbol, output_dir="./yfinance_dataset", train_ratio=0.8):
    """
    Download data for a given symbol and split it into train/test sets.
    
    Args:
        symbol (str): Stock symbol (e.g., 'SPY')
        output_dir (str): Directory to save the data
        train_ratio (float): Ratio of data to use for training (0.8 = 80% training, 20% testing)
    """
    # Create output directories
    train_dir = os.path.join(output_dir, "split", "train")
    test_dir = os.path.join(output_dir, "split", "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Download data
    print(f"Downloading data for {symbol}...")
    ticker = yf.Ticker(symbol)
    
    try:
        # Get maximum available data
        data = ticker.history(period="max")
        
        if len(data) == 0:
            print(f"No data available for {symbol}, skipping...")
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
        
        # Sort by date
        data = data.sort_values('Date')
        
        # Calculate split index
        split_idx = int(len(data) * train_ratio)
        
        # Split into train and test
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        # Save files
        train_file = os.path.join(train_dir, f"{symbol}.csv")
        test_file = os.path.join(test_dir, f"{symbol}.csv")
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        print(f"Data saved for {symbol}")
        print(f"Training data range: {train_data['Date'].min()} to {train_data['Date'].max()}")
        print(f"Testing data range: {test_data['Date'].min()} to {test_data['Date'].max()}")
        print(f"Total rows: {len(data)}, Training rows: {len(train_data)}, Testing rows: {len(test_data)}")
    
    except Exception as e:
        print(f"Error downloading {symbol}: {str(e)}") 