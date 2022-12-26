import os
import numpy as np
from model_generator import ModelGenerator
from config import Constants, Variables
from custom_loss import CustomLoss
import tensorflow as tf
from random import shuffle
import cv2


class Train:
    def train(self):
        model_generator = ModelGenerator()
        model = model_generator.create_autoencoder()
        # model.Complie(optimizer='adam', loss='binary_crossentropy')
        if Variables.init_weight_file_address != '' and Variables.init_weight_file_address is not None:
            weight_address = Constants.weight_pass + Variables.model_name + '/' + Variables.init_weight_file_address
            model.load_weights(weight_address)
        loss_function = CustomLoss()

        file_path = Constants.train_path_noisy
        file_names = os.listdir(file_path)
        shuffle(file_names)

        step_per_epoch = len(file_names) // Variables.batch_size

        for epoch in range(Variables.start_epoch, Variables.end_epoch):
            # later add some condition for computing learning_rate and optimizer
            optimizer = self._get_optimizer(lr=5e-3)
            total_loss = 0.0
            for batch_index in range(step_per_epoch):
                source_images, target_images = self._get_batch_samples(batch_index, file_names)
                source_images = tf.cast(source_images, tf.float32)
                target_images = tf.cast(target_images, tf.float32)
                loss = self._train_model(source_images, target_images, model, loss_function, optimizer)
                total_loss += loss
                tf.print("EPOCH:", str(epoch), " ->STEP:", str(batch_index) + '/' + str(step_per_epoch), ' -> : LOSS:',
                         tf.reduce_mean(loss))
            self._save_weights(epoch, model, total_loss, step_per_epoch)

    def _train_model(self, source_images, target_images, model, loss_function, optimizer):
        with tf.GradientTape() as tape:
            model_prediction = model(source_images, training=True)
            loss = loss_function.calculate_mse(x_pr=model_prediction, x_gt=target_images)

        model_gradient = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))
        return loss

    def _get_optimizer(self, lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-5):
        try:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, weight_decay=decay)
        except:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
        return optimizer

    def _get_batch_samples(self, batch_index, file_names):
        files = file_names[batch_index * Variables.batch_size:(batch_index + 1) * Variables.batch_size]
        source_batch = []
        target_batch = []
        for file in files:
            info = np.load(Constants.train_path_noisy + file, allow_pickle=True)
            info = info.item()
            if Variables.model_name == 'full_denoising':
                noisy = info['full_noisy']
                clear = info['image']
                source_batch.append(noisy)
                target_batch.append(clear)

            elif Variables.model_name == 'central_denoising':
                noisy = info['patched_noisy']
                clear = info['patched_non_noisy']
                [source_batch.append(patch) for patch in noisy]
                [target_batch.append(patch) for patch in clear]

            elif Variables.model_name == 'central_reconstruction':
                noisy = info['patched_destroyed']
                clear = info['patched_non_destroyed']
                [source_batch.append(patch) for patch in noisy]
                [target_batch.append(patch) for patch in clear]

        source_batch = np.array(source_batch)
        source_batch = source_batch / 255.0
        source_batch = np.expand_dims(source_batch, axis=-1)

        target_batch = np.array(target_batch)
        target_batch = target_batch / 255.0
        target_batch = np.expand_dims(target_batch, axis=-1)

        return source_batch, target_batch

    def _save_weights(self, epoch, model, total_loss, step_per_epoch):
        save_path = Constants.weight_path
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        save_path += Variables.model_name + '/'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        total_loss = tf.reduce_mean(total_loss)
        mse_per_epoch = total_loss / step_per_epoch
        mse_per_epoch = mse_per_epoch.numpy()
        mse_per_epoch = float(int(mse_per_epoch * 10 ** 6)) / 10 ** 6
        save_name = save_path + 'epo_' + str(epoch) + '__mse_' + str(mse_per_epoch) + '.h5'
        model.save(save_name)


