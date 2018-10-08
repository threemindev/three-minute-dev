from keras.models import Model
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Dense
from keras.layers import Conv2D, UpSampling2D
from .ragan_components import leaky_relu


def build_mnist_generator(random_dims: int):
    x = inp = Input(shape=(random_dims,))

    x = Dense(units=(7 * 7) * 32, activation=leaky_relu, name='generator_dense_1')(x)
    x = Reshape((7, 7, 32))(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=leaky_relu, name='generator_width_14_activated')(UpSampling2D(size=(2, 2))(x))
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=leaky_relu, name='generator_width_28_activated')(UpSampling2D(size=(2, 2))(x))
    x = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation=leaky_relu, name='generator_last_conv_1')(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation=leaky_relu, name='generator_last_conv_2')(x)
    out = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid', name='generator_out')(x)

    generator = Model(inputs=[inp], outputs=[out], name='generator')
    return generator
