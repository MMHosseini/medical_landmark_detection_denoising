from model_generator import ModelGenerator
from data_helper import DataHelper


def create_dataset():
    dhl = DataHelper()
    dhl.make_dataset()


def train_model():
    generator = ModelGenerator()
    generator.create_autoencoder()


if __name__ == '__main__':
    create_dataset()
    # train_model()

