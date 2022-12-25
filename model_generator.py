import tensorflow as tf
from keras import Input, layers, Model
from config import Constants, Variables


class ModelGenerator:
    def create_autoencoder(self):
        models = Variables.models['full_denoising', 'central_denoising', 'central_reconstruction']
        model_name = Variables.model_name
        if model_name == 'full_denoising':
            input_shape = (Variables.image_input_size_height, Variables.image_input_size_width, 1)
        elif model_name == 'central_denoising' or model_name == 'central_reconstruction':
            check_height = Variables.image_input_size_height % Variables.num_batches[0] == 0
            check_width = Variables.image_input_size_width % Variables.num_batches[1] == 0
            if not (check_height and check_width):
                print('Error in creating autoencoder: patch size is not consistent')
                return
            height = Variables.image_input_size_height / Variables.num_batches[0]
            width = Variables.image_input_size_width / Variables.num_batches[1]
            input_shape = (height, width, 3)
        input_img = Input(shape=input_shape, name='input')
        encode = self.__encoder(input_img)
        decode = self.__decoder(encode)
        autoencoder = Model(input_img, decode, name='auto-encoder')
        autoencoder.summary()
        return autoencoder

    def __encoder(self, x):
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='enc-conv1')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='enc-pooling1')(x)
        x = layers.BatchNormalization(name='enc-normalization1')(x)

        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='enc-conv2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='enc-pooling2')(x)
        x = layers.BatchNormalization(name='enc-normalization2')(x)

        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='enc-conv3')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='enc-pooling3')(x)
        x = layers.BatchNormalization(name='enc-normalization3')(x)
        return x

    def __decoder(self, x):
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='dec_conv1')(x)
        x = layers.UpSampling2D(size=(2, 2), name='dec-upsampling1')(x)
        x = layers.BatchNormalization(name='dec-normalization1')(x)

        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='dec-conv2')(x)
        x = layers.UpSampling2D(size=(2, 2), name='dec-upsampling2')(x)
        x = layers.BatchNormalization(name='dec-normalization2')(x)

        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='dec-conv3')(x)
        x = layers.UpSampling2D(size=(2, 2), name='dec-upsampling3')(x)
        x = layers.BatchNormalization(name='dec-normalization3')(x)

        x = layers.Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same', name='dec-conv4')(x)
        x = layers.BatchNormalization(name='dec-normalization4')(x)
        return x





