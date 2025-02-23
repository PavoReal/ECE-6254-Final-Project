import argparse
import os
from ece6254 import train, test, dataset

def print_arch_list():
    longest_name = 0
    longet_desc  = 0

    for arch in train.model_arch:
        longest_name = max(longest_name, len(arch["name"]))
        longet_desc  = max(longet_desc, len(arch["desc"]))

    longest_name = max(14, longest_name)

    total_width = longest_name + longet_desc + 4

    print("~" * total_width)

    print(f'{"name".ljust(longest_name)} -- {"description".ljust(longet_desc)}')
    print(f'{"-" * (longest_name + longet_desc + 4)}')

    for arch in train.model_arch:
        
        default = "";
        if arch == train.model_arch[0]:
            default = " (default)"

        print(f'{(arch["name"] + default).ljust(longest_name)} -- {arch["desc"].ljust(longet_desc)}')

    print("~" * total_width)

def print_dataset_list(folder):
    always_exclude = ['symbols_valid_meta.csv']

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file in always_exclude:
                continue;

            print(f'{file}')


def run_main():
    parser = argparse.ArgumentParser(description='CLI interface for training and testing models.')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help (train or test)')

    # Train command
    train_parser = subparsers.add_parser('train',                                                   help='Train a model')
    train_parser.add_argument('-m', '--model_file', type=str,  required=True,                       help='Path to the model file (without extension) to save or load the model')
    train_parser.add_argument('-d', '--data_name',  type=str,  required=True,                       help='Name of the dataset item, use command dataset_list for a complete list')
    train_parser.add_argument('--data_dir',         type=str,  default="./dataset",                 help='Override the default dataset dir of ./dataset')
    train_parser.add_argument('-f', '--features',   nargs='+', default=['Close', 'Volume'],                   help='Features to train on')
    train_parser.add_argument('-a', '--model_arch', type=str,  default=train.model_arch[0]["name"], help='Change the model architecture, use command arch_list for a complete list')
    train_parser.add_argument('-s', '--seq_length', type=int,  default=30,                          help='Sequence length for training')
    train_parser.add_argument('-e', '--epochs',     type=int,  default=80,                          help='Number of epochs')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test a model')
    test_parser.add_argument('-m', '--model_path', type=str, required=True,       help='Path to the model file (without extension)')
    test_parser.add_argument('-d', '--data_name',  type=str, required=True,       help='Name of the dataset item, use command dataset_list for a complete list')
    test_parser.add_argument('--data_dir',         type=str, default="./dataset", help='Override the default dataset dir of ./dataset')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two already trained models')
    compare_parser.add_argument('-a', '--model_path_a', type=str, required=True,       help='Path to model file 1 (without extension)')
    compare_parser.add_argument('-b', '--model_path_b', type=str, required=True,       help='Path to model file 2 (without extension)')
    compare_parser.add_argument('-d', '--data_name',     type=str, required=True,       help='Name of dataset to use, use command dataset_list for a complete list of available datasets')
    compare_parser.add_argument('--data_dir',           type=str, default="./dataset", help='Override the default dataset dir of ./dataset')

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
        test.compare_main(model_path_a=args.model_path_a, model_path_b=args.model_path_b, data_name=args.data_name, data_dir=args.data_dir)
    elif args.command == 'arch_list':
        print_arch_list()
    elif args.command == 'dataset_list':
        print_dataset_list(args.path)
    else:
        parser.print_help()

if __name__ == '__main__':
    run_main()
