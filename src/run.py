import argparse
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

def run_main():
    parser = argparse.ArgumentParser(description='CLI interface for training and testing models.')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help (train or test)')

    # Train command parser
    train_parser = subparsers.add_parser('train',                                                   help='Train a model')
    train_parser.add_argument('-m', '--model_file', type=str,  required=True,                       help='Path to the model file (without extension) to save or load the model')
    train_parser.add_argument('-d', '--data_file',  type=str,  required=True,                       help='Path to the training data file')
    train_parser.add_argument('-t', '--test_data',  type=str,  required=True,                       help='Path to the testing data file')
    train_parser.add_argument('-f', '--features',   nargs='+', default=['Close'],                   help='Features to train on')
    train_parser.add_argument('-a', '--model_arch', type=str,  default=train.model_arch[0]["name"], help='Change the model architecture, use command arch_list for a complete list')
    train_parser.add_argument('-s', '--seq_length', type=int,  default=30,                          help='Sequence length for training')
    train_parser.add_argument('-e', '--epochs',     type=int,  default=80,                          help='Number of epochs')

    # Test command parser
    test_parser = subparsers.add_parser('test', help='Test a model')
    test_parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the model file (without extension) to load the model')
    test_parser.add_argument('-d', '--data_file',  type=str, required=True, help='Path to the testing data file')

    download_parser = subparsers.add_parser('download', help='Download the dataset')
    download_parser.add_argument('-p', '--path', type=str, required=True, help='Path to save the dataset')

    arch_list_parser = subparsers.add_parser('arch-list', help='List all avaiable model architectures')

    args = parser.parse_args()

    if args.command == 'train':
        train.train_main(model_file_path=args.model_file, train_file_path=args.data_file, 
            test_file_path=args.test_data, features=args.features, 
            seq_length=args.seq_length, epochs=args.epochs, 
            model_arch=train.get_model_arch(args.model_arch))
    elif args.command == 'test':
        test.test_main(args.model_path, args.data_file)
    elif args.command == 'download':
        dataset.download_and_save(args.path)
    elif args.command == 'arch-list':
        print_arch_list();
    else:
        parser.print_help()

if __name__ == '__main__':
    run_main()
