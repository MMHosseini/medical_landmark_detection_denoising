from train import Train
from data_helper import DataHelper


def create_dataset():
    data_helper = DataHelper()
    data_helper.make_dataset()


def train_model():
    trainer = Train()
    trainer.train()


if __name__ == '__main__':
    # create_dataset()
    train_model()

