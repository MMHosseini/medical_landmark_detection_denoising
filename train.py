import os

import numpy as np

from model_generator import ModelGenerator
from config import Constants, Variables
import tensorflow as tf
import keras


class Train:
    def train(self):
        model_generator = ModelGenerator()
        model = model_generator.create_autoencoder()
        model.Complie(optimizer='adam', loss='binary_crossentropy')
        if Variables.init_weight_file_address is not '' and Variables.init_weight_file_address is not None:
            model.load_weights(Constants.weight_pass + Variables.model_name + '/' + Variables.init_weight_file_address)

        image_list = os.listdir(Constants.train_path)
        num_images = len(image_list)

        epochs = Variables.num_epochs
        batches = Variables.num_batches
        for epoch in epochs:
            steps = np.ceil(num_images/batches)
            for step in steps:
                print('a')

    def train_step(self, step):
        print(step)


