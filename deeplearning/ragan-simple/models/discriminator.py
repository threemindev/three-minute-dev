from .ragan_components import leaky_relu

from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense

from typing import Collection


def build_mnist_discriminator(image_shape: Collection[int]):
    x = inp = Input(shape=image_shape)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation=leaky_relu, strides=(2, 2), name='discriminator_conv_0')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation=leaky_relu, name='discriminator_conv_1')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='valid', activation=leaky_relu, name='discriminator_conv_2')(x)
    x = Conv2D(filters=4, kernel_size=(3, 3), padding='valid', activation=leaky_relu, name='discriminator_conv_3')(x)
    x = Flatten()(x)

    x = Dense(32, activation=leaky_relu, name='discriminator_dense_0')(x)
    out = Dense(1, activation='linear', name='discriminator_dense_out')(x)

    discriminator = Model(inputs=[inp], outputs=[out], name='discriminator')
    return discriminator


def build_cifar10_discriminator(image_shape: Collection[int]):
    x = inp = Input(shape=image_shape)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation=leaky_relu, strides=(2, 2), name='discriminator_conv_0')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation=leaky_relu, strides=(2, 2), name='discriminator_conv_1')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='valid', activation=leaky_relu, name='discriminator_conv_2')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='valid', activation=leaky_relu, name='discriminator_conv_3')(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), padding='valid', activation=leaky_relu, name='discriminator_conv_4')(x)
    x = Flatten()(x)

    x = Dense(64, activation=leaky_relu, name='discriminator_dense_0')(x)
    x = Dense(32, activation=leaky_relu, name='discriminator_dense_1')(x)
    out = Dense(1, activation='linear', name='discriminator_dense_out')(x)

    discriminator = Model(inputs=[inp], outputs=[out], name='discriminator')
    return discriminator
