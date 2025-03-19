import argparse
import os
from ece6254 import train, test, dataset, models, dataset_yfinance

def run_main():
    parser = argparse.ArgumentParser(description='CLI interface for training and testing models.')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help (train or test)')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('-m', '--model_file', type=str, required=True, help='Path to the model file (without extension) to save or load the model')
    train_parser.add_argument('-d', '--data_name', type=str, required=True, help='Name of the dataset item, use command dataset_list for a complete list')
    train_parser.add_argument('--data_dir', type=str, default="./dataset", help='Override the default dataset dir of ./dataset')
    train_parser.add_argument('--data_source', type=str, choices=['kaggle', 'yfinance'], default='kaggle', help='Choose data source: kaggle (original dataset) or yfinance (fresh data)')
    train_parser.add_argument('-f', '--features', nargs='+', default=['Close', 'Open', 'High', 'Low'], help='Features to train on')
    train_parser.add_argument('-a', '--model_arch', type=str, default=models.model_arch[0]["name"], help='Change the model architecture, use command arch_list for a complete list')
    train_parser.add_argument('-s', '--seq_length', type=int, default=30, help='Sequence length for training')
    train_parser.add_argument('-e', '--epochs', type=int, default=80, help='Number of epochs')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test a model')
    test_parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model file (without extension)')
    test_parser.add_argument('-d', '--data_name', type=str, required=True, help='Name of the dataset item, use command dataset_list for a complete list')
    test_parser.add_argument('--data_dir', type=str, default="./dataset", help='Override the default dataset dir of ./dataset')
    test_parser.add_argument('--data_source', type=str, choices=['kaggle', 'yfinance'], default='kaggle', help='Choose data source: kaggle (original dataset) or yfinance (fresh data)')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple trained models')
    compare_parser.add_argument('-m', '--model_paths', type=str, nargs='+', required=True, help='List of model file paths (without extension) to compare')
    compare_parser.add_argument('-d', '--data_name', type=str, required=True, help='Name of dataset to use, use command dataset_list for a complete list of available datasets')
    compare_parser.add_argument('--data_dir', type=str, default="./dataset", help='Override the default dataset dir of ./dataset')
    compare_parser.add_argument('--data_source', type=str, choices=['kaggle', 'yfinance'], default='kaggle', help='Choose data source: kaggle (original dataset) or yfinance (fresh data)')

    # Download command (Kaggle dataset)
    download_parser = subparsers.add_parser('download', help='Download the Kaggle dataset')
    download_parser.add_argument('-p', '--path', type=str, default="./dataset", help='Path to save the dataset, defaults to ./dataset')

    # Download fresh data command (yfinance)
    yfinance_parser = subparsers.add_parser('download_yfinance', help='Download fresh data using yfinance')
    yfinance_parser.add_argument('--output', default='./yfinance_dataset', help='Output directory')
    yfinance_parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of data to use for training')
    yfinance_parser.add_argument('--symbols', nargs='+', help='Optional: Specific stock symbols to download. If not provided, will download all symbols from current dataset.')
    yfinance_parser.add_argument('--dataset-dir', default='./dataset', help='Directory containing the current dataset')

    # Arch list command
    arch_list_parser = subparsers.add_parser('arch_list', help='List all available model architectures')

    # Dataset list command
    dataset_list_parser = subparsers.add_parser('dataset_list', help='List all available datasets.')
    dataset_list_parser.add_argument('-p', '--path', type=str, default="./dataset", help='Path to dataset dir, defaults to ./dataset')
    dataset_list_parser.add_argument('--data_source', type=str, choices=['kaggle', 'yfinance'], default='kaggle', help='Choose data source: kaggle (original dataset) or yfinance (fresh data)')

    args = parser.parse_args()

    if args.command == 'train':
        data_dir = "./yfinance_dataset" if args.data_source == 'yfinance' else args.data_dir
        train.train_main(model_file_path=args.model_file, data_name=args.data_name, data_dir=data_dir, features=args.features, 
            seq_length=args.seq_length, epochs=args.epochs, model_arch=train.get_model_arch(args.model_arch))
    elif args.command == 'test':
        data_dir = "./yfinance_dataset" if args.data_source == 'yfinance' else args.data_dir
        test.test_main(model_path=args.model_path, data_name=args.data_name, data_dir=data_dir)
    elif args.command == 'download':
        dataset.download_and_save(args.path)
    elif args.command == 'download_yfinance':
        if not args.symbols:
            symbols = dataset_yfinance.get_all_symbols(args.dataset_dir)
            print(f"Found {len(symbols)} symbols in current dataset")
        else:
            symbols = args.symbols
            print(f"Using {len(symbols)} provided symbols")
        
        for symbol in symbols:
            try:
                dataset_yfinance.download_and_split_data(symbol, args.output, args.train_ratio)
            except Exception as e:
                print(f"Error downloading {symbol}: {str(e)}")
    elif args.command == 'compare':
        data_dir = "./yfinance_dataset" if args.data_source == 'yfinance' else args.data_dir
        test.compare_main(model_paths=args.model_paths, data_name=args.data_name, data_dir=data_dir)
    elif args.command == 'arch_list':
        models.print_arch_list()
    elif args.command == 'dataset_list':
        data_dir = "./yfinance_dataset" if args.data_source == 'yfinance' else args.path
        dataset.print_dataset_list(data_dir)
    else:
        parser.print_help()

if __name__ == '__main__':
    run_main()
