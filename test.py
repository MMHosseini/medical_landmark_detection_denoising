from config import Constants, Variables
from model_generator import ModelGenerator
from custom_loss import CustomLoss
from data_helper import DataHelper
import os
import numpy as np
import cv2
from tqdm import tqdm


class Test:
    def evaluate_test_samples(self):
        model_generator = ModelGenerator()
        model = model_generator.create_autoencoder()

        weight_address = Constants.weight_path + Variables.model_name + '/' + Variables.test_model_weight
        model.load_weights(weight_address)

        loss_function = CustomLoss()
        data_helper = DataHelper()

        file_path = Constants.test_path_noisy
        file_names = os.listdir(file_path)
        num_samples = len(file_names)

        loss = 0
        for file_name in tqdm(file_names):
            file_adr = file_path + file_name
            info = np.load(file_adr, allow_pickle=True).item()

            if Variables.model_name == 'full_denoising':
                clear = info['image']
                noisy = info['full_noisy']
                clear = np.expand_dims(clear, axis=0)
                noisy = np.expand_dims(noisy, axis=0)
            elif Variables.model_name == 'central_denoising':
                clear = info['patched_non_noisy']
                noisy = info['patched_noisy']
            elif Variables.model_name == 'central_reconstruction':
                clear = info['patched_non_destroyed']
                noisy = info['patched_destroyed']

            clear = np.expand_dims(clear, axis=-1)
            noisy = np.expand_dims(noisy, axis=-1)

            clear = clear / 255.0
            noisy = noisy / 255.0

            prediction = model.predict(clear, verbose=0)
            loss += loss_function.calculate_mse(x_gt=clear, x_pr=prediction)

            clear = clear * 255
            noisy = noisy * 255
            prediction = prediction * 255

            if Variables.model_name == 'full_denoising':
                recovered = prediction[0, :, :, 0]
            elif Variables.model_name == 'central_denoising' or Variables.model_name == 'central_reconstruction':
                recovered, uncovered_area = data_helper.depatch_image(prediction)
                recovered[uncovered_area == 1] = info['image'][uncovered_area == 1]

            # cv2.imwrite('/home/mehdi/Desktop/temp/reconstructed.jpg', reconstructed)
            # cv2.imwrite('/home/mehdi/Desktop/temp/clear.jpg', info['image'])

            if Variables.model_name == 'full_denoising':
                info['full_noisy_reconstructed'] = recovered
            elif Variables.model_name == 'central_denoising':
                info['patched_noisy_reconstructed'] = recovered
            elif Variables.model_name == 'central_reconstruction':
                info['patched_destroyed_reconstructed'] = recovered

            np.save(file_adr, info)

        avg_loss = loss / num_samples
        avg_loss = float(int(avg_loss*10e6)) / 10e6
        print('Model:', Variables.model_name, ' -> Num Samples:', str(num_samples), ' -> Average loss:', str(avg_loss))

    def evaluate_train_samples(self):
        model_generator = ModelGenerator()
        model = model_generator.create_autoencoder()

        weight_address = Constants.weight_path + Variables.model_name + '/' + Variables.test_model_weight
        model.load_weights(weight_address)

        loss_function = CustomLoss()
        data_helper = DataHelper()

        file_path = Constants.train_path_noisy
        file_names = os.listdir(file_path)
        num_samples = len(file_names)

        loss = 0
        for file_name in tqdm(file_names):
            file_adr = file_path + file_name
            info = np.load(file_adr, allow_pickle=True).item()

            if Variables.model_name == 'full_denoising':
                clear = info['image']
                noisy = info['full_noisy']
                clear = np.expand_dims(clear, axis=0)
                noisy = np.expand_dims(noisy, axis=0)
            elif Variables.model_name == 'central_denoising':
                clear = info['patched_non_noisy']
                noisy = info['patched_noisy']
            elif Variables.model_name == 'central_reconstruction':
                clear = info['patched_non_destroyed']
                noisy = info['patched_destroyed']

            clear = np.expand_dims(clear, axis=-1)
            noisy = np.expand_dims(noisy, axis=-1)

            clear = clear / 255.0
            noisy = noisy / 255.0

            prediction = model.predict(clear, verbose=0)
            loss += loss_function.calculate_mse(x_gt=clear, x_pr=prediction)

            clear = clear * 255
            noisy = noisy * 255
            prediction = prediction * 255

            if Variables.model_name == 'full_denoising':
                recovered = prediction[0, :, :, 0]
            elif Variables.model_name == 'central_denoising' or Variables.model_name == 'central_reconstruction':
                recovered, uncovered_area = data_helper.depatch_image(prediction)
                recovered[uncovered_area == 1] = info['image'][uncovered_area == 1]

            # cv2.imwrite('/home/mehdi/Desktop/temp/reconstructed.jpg', reconstructed)
            # cv2.imwrite('/home/mehdi/Desktop/temp/clear.jpg', info['image'])

            if Variables.model_name == 'full_denoising':
                info['full_noisy_reconstructed'] = recovered
            elif Variables.model_name == 'central_denoising':
                info['patched_noisy_reconstructed'] = recovered
            elif Variables.model_name == 'central_reconstruction':
                info['patched_destroyed_reconstructed'] = recovered

            np.save(file_adr, info)

        avg_loss = loss / num_samples
        avg_loss = float(int(avg_loss*10e6)) / 10e6
        print('Model:', Variables.model_name, ' -> Num Samples:', str(num_samples), ' -> Average loss:', str(avg_loss))

