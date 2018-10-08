from keras.models import Model
from keras.layers import Input
from keras.layers import Subtract
from keras.layers import Lambda
from keras.layers import Activation

from keras.activations import relu
from keras.optimizers import RMSprop, Adam

import keras.backend as K


def leaky_relu(x):
    return relu(x, alpha=.2)


class BatchAverageLayer(Lambda):
    """
    Repeat the average along a batch, keeping dims
    """
    def __init__(self, batch_size: int, *args, **kwargs):
        if batch_size <= 0:
            raise ValueError('batch_size should be positive integer')

        def _batch_average(x):
            return K.repeat_elements(K.mean(x, axis=0, keepdims=True), int(batch_size), axis=0)
        super(BatchAverageLayer, self).__init__(function=_batch_average, *args, **kwargs)


def build_compiled_ragan_trainers(generator: Model, discriminator: Model, batch_size: int, use_rmsprop: bool=True):
    generator_old_trainable = generator.trainable
    discriminator_old_trainable = discriminator.trainable

    generator.trainable = False
    discriminator.trainable = True
    random_inp = Input(batch_shape=[batch_size, int(generator.get_input_shape_at(0)[1])]) # Random z
    image_inp = Input(batch_shape=[batch_size] + [int(item) for item in discriminator.get_input_shape_at(0)[1:]]) # Real Images

    discriminator_trainer_real = Model(inputs=[image_inp, random_inp], outputs=[Activation('sigmoid')(Subtract()([discriminator(image_inp), BatchAverageLayer(batch_size=batch_size)(discriminator(generator(random_inp)))]))],name='discriminator_trainer_real')
    discriminator_trainer_fake = Model(inputs=[image_inp, random_inp], outputs=[Activation('sigmoid')(Subtract()([discriminator(generator(random_inp)), BatchAverageLayer(batch_size=batch_size)(discriminator(image_inp))]))],name='discriminator_trainer_fake')
    discriminator_trainer_real.compile(optimizer=RMSprop(lr=1e-4) if use_rmsprop else Adam(lr=1e-4, beta_1=.5), loss='binary_crossentropy')
    discriminator_trainer_fake.compile(optimizer=RMSprop(lr=1e-4) if use_rmsprop else Adam(lr=1e-4, beta_1=.5), loss='binary_crossentropy')

    generator.trainable = True
    discriminator.trainable = False
    generator_trainer_real = Model(inputs=[image_inp, random_inp], outputs=[Activation('sigmoid')(Subtract()([discriminator(image_inp), BatchAverageLayer(batch_size=batch_size)(discriminator(generator(random_inp)))]))], name='generator_trainer_real')
    generator_trainer_fake = Model(inputs=[image_inp, random_inp], outputs=[Activation('sigmoid')(Subtract()([discriminator(generator(random_inp)), BatchAverageLayer(batch_size=batch_size)(discriminator(image_inp))]))], name='generator_trainer_fake')
    generator_trainer_real.compile(optimizer=RMSprop(lr=1e-4) if use_rmsprop else Adam(lr=1e-4, beta_1=.5), loss='binary_crossentropy')
    generator_trainer_fake.compile(optimizer=RMSprop(lr=1e-4) if use_rmsprop else Adam(lr=1e-4, beta_1=.5), loss='binary_crossentropy')

    generator.trainable = generator_old_trainable
    discriminator.trainable = discriminator_old_trainable

    return discriminator_trainer_real, discriminator_trainer_fake, generator_trainer_real, generator_trainer_fake
