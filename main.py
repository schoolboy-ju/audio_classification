from data_loader.common_data_loader import CommonDataLoader
from models.conv_1d import Conv1D
from trainers.common_trainer import CommonTrainer
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

    print("Create the data generator.")
    data_loader = CommonDataLoader(config)

    data_loader.get_train_data()

    print("Create the model.")
    model = Conv1D(config)

    print("Create the trainer")
    trainer = CommonTrainer(model.model,
                            data_loader.get_train_data(),
                            data_loader.get_test_data(),
                            config)

    print("Start training the model.")
    trainer.train()


if __name__ == '__main__':
    main()
