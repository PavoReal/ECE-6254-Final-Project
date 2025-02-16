import argparse
from ece6254 import train, test, dataset

def run_main():
    parser = argparse.ArgumentParser(description='CLI interface for training and testing models.')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help (train or test)')

    # Train command parser
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('-m', '--model_file', type=str, required=True, help='Path to the model file (without extension) to save or load the model')
    train_parser.add_argument('-d', '--data_file', type=str, required=True, help='Path to the training data file')
    train_parser.add_argument('-f', '--features', nargs='+', default=['Close'], help='Features to train on')
    train_parser.add_argument('-s', '--seq_length', type=int, default=30, help='Sequence length for training')
    train_parser.add_argument('-e', '--epochs', type=int, default=80, help='Number of epochs')

    # Test command parser
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model file (without extension) to load the model')

    download_parser = subparsers.add_parser('download', help='Download the dataset')
    download_parser.add_argument('-p', '--path', type=str, required=True, help='Path to save the dataset')

    args = parser.parse_args()

    if args.command == 'train':
        train.train_main(args.model_file, args.data_file, features=args.features, seq_length=args.seq_length, epochs=args.epochs)
    elif args.command == 'test':
        test.test_main(args.model_path)
    elif args.command == 'download':
        dataset.download_and_save(args.path)
    else:
        parser.print_help()

if __name__ == '__main__':
    run_main()