import argparse
import os
from ece6254 import train, test, dataset, models

def run_main():
    parser = argparse.ArgumentParser(description='CLI interface for training and testing models.')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help (train or test)')

    # Train command
    train_parser = subparsers.add_parser('train',                                                   help='Train a model')
    train_parser.add_argument('-m', '--model_file', type=str,  required=True,                       help='Path to the model file (without extension) to save or load the model')
    train_parser.add_argument('-d', '--data_name',  type=str,  required=True,                       help='Name of the dataset item, use command dataset_list for a complete list')
    train_parser.add_argument('--data_dir',         type=str,  default="./dataset",                 help='Override the default dataset dir of ./dataset')
    train_parser.add_argument('-f', '--features',   nargs='+', default=['Close', 'Open', 'High', 'Low'],           help='Features to train on')
    train_parser.add_argument('-a', '--model_arch', type=str,  default= models.model_arch[0]["name"], help='Change the model architecture, use command arch_list for a complete list')
    train_parser.add_argument('-s', '--seq_length', type=int,  default=30,                          help='Sequence length for training')
    train_parser.add_argument('-e', '--epochs',     type=int,  default=80,                          help='Number of epochs')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test a model')
    test_parser.add_argument('-m', '--model_path', type=str, required=True,       help='Path to the model file (without extension)')
    test_parser.add_argument('-d', '--data_name',  type=str, required=True,       help='Name of the dataset item, use command dataset_list for a complete list')
    test_parser.add_argument('--data_dir',         type=str, default="./dataset", help='Override the default dataset dir of ./dataset')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple trained models')
    compare_parser.add_argument('-m', '--model_paths', type=str, nargs='+', required=True,  help='List of model file paths (without extension) to compare')
    compare_parser.add_argument('-d', '--data_name',   type=str, required=True,            help='Name of dataset to use, use command dataset_list for a complete list of available datasets')
    compare_parser.add_argument('--data_dir',          type=str, default="./dataset",      help='Override the default dataset dir of ./dataset')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download the dataset')
    download_parser.add_argument('-p', '--path', type=str, default="./dataset", help='Path to save the dataset, defaults to ./dataset')

    # Arch list command
    arch_list_parser = subparsers.add_parser('arch_list',    help='List all available model architectures')

    # Dataset list commadn
    dataset_list_parser = subparsers.add_parser('dataset_list', help='List all available datasets. ')
    dataset_list_parser.add_argument('-p', '--path', type=str, default="./dataset", help='Path to dataset dir, defaults to ./dataset')

    args = parser.parse_args()

    if args.command == 'train':
        train.train_main(model_file_path=args.model_file, data_name=args.data_name, data_dir=args.data_dir, features=args.features, 
            seq_length=args.seq_length, epochs=args.epochs, model_arch=train.get_model_arch(args.model_arch))
    elif args.command == 'test':
        test.test_main(model_path=args.model_path, data_name=args.data_name, data_dir=args.data_dir)
    elif args.command == 'download':
        dataset.download_and_save(args.path)
    elif args.command == 'compare':
        test.compare_main(model_paths=args.model_paths, data_name=args.data_name, data_dir=args.data_dir)
    elif args.command == 'arch_list':
        models.print_arch_list()
    elif args.command == 'dataset_list':
        dataset.print_dataset_list(args.path)
    else:
        parser.print_help()

if __name__ == '__main__':
    run_main()
