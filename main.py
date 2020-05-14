from data_loader.data_loader import DataLoader
from utils.configs import process_config
from utils.args import get_args


def main():
    # capture the configs path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.configs)
    except:
        print("missing or invalid arguments")
        exit(0)

    print('Create the data generator.')
    data_loader = DataLoader(config)


if __name__ == '__main__':
    main()
