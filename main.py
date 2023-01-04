from train import Train
from data_helper import DataHelper
from test import Test

def create_dataset():
    data_helper = DataHelper()
    data_helper.make_dataset()


def train_model():
    trainer = Train()
    trainer.train()


def test_model():
    tester = Test()
    tester.evaluate_train_samples()
    tester.evaluate_test_samples()


if __name__ == '__main__':
    # create_dataset()
    # train_model()
    test_model()

