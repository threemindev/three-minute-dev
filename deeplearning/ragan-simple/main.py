from models import build_mnist_generator, build_cifar10_generator
from models import build_mnist_discriminator, build_cifar10_discriminator
from models import build_compiled_ragan_trainers
from models import leaky_relu

import keras.datasets

from keras.models import Model, load_model

from PIL import Image
import numpy as np

from typing import Collection
from tqdm import tqdm
import os
from time import time


#
def normalized(x: np.ndarray):
    return x.astype('float32') / 255.


def denormalized(x: np.ndarray):
    return (x * 255.).astype('u1')


def normalized_to_gray_image_array(x: np.ndarray):
    return np.squeeze(denormalized(x), axis=-1)


def gray_image_array_to_normalized(x: np.ndarray):
    return np.expand_dims(normalized(x), axis=-1)


#
def train(real_data:np.ndarray, generator: Model, discriminator: Model, target_epochs: int, batch_size: int, train_label: str, random_dims: int, image_shape: Collection[int], result_dir_path: str='result', load_epoch=None):
    if load_epoch is not None:
        raise NotImplementedError('load_epoch is not supported yet')
    if not isinstance(real_data, np.ndarray):
        raise TypeError('real_data should be Grayscale or RGB images dataset')
    if real_data.ndim != 3 and real_data.ndim != 4:
        raise ValueError('real_data should be Grayscale or RGB images dataset')
    if batch_size <= 0:
        raise ValueError('batch_size should be a positive integer')
    generator_input_shape = [int(item) for item in generator.get_input_shape_at(0)[1:]]
    if len(generator_input_shape) != 1 or random_dims != generator_input_shape[0]:
        raise ValueError('wrong random_dims')
    discriminator_input_shape = [int(item) for item in discriminator.get_input_shape_at(0)[1:]]
    if len(discriminator_input_shape) != 3 or discriminator_input_shape != list(image_shape):
        raise ValueError('image_shape is not a shape with size and channels')

    generator.summary()
    discriminator.summary()

    discriminator_trainer_real, discriminator_trainer_fake, generator_trainer_real, generator_trainer_fake = build_compiled_ragan_trainers(generator=generator,
                                                                                                                                           discriminator=discriminator,
                                                                                                                                           batch_size=batch_size,
                                                                                                                                           use_rmsprop=True,
                                                                                                                                           learning_rate=1e-3)

    #
    epoch_tqdm =tqdm(range(target_epochs), desc='epochs', position=0)
    for _ in epoch_tqdm:
        losses = [0.] * 4
        np.random.shuffle(real_data)
        batch_tqdm = tqdm(range(len(real_data) // batch_size), desc='batches', position=1)
        for batch_index in batch_tqdm:
            batch_real = normalized(real_data[batch_index * batch_size:(batch_index + 1) * batch_size])
            batch_random = np.random.randn(4, batch_size, random_dims)

            losses[0] = discriminator_trainer_real.train_on_batch(x=[batch_real, batch_random[0]], y=np.ones(shape=(batch_size, 1)))
            losses[1] = discriminator_trainer_fake.train_on_batch(x=[batch_real, batch_random[1]], y=np.zeros(shape=(batch_size, 1)))

            losses[2] = generator_trainer_real.train_on_batch(x=[batch_real, batch_random[2]], y=np.zeros(shape=(batch_size, 1)))
            losses[3] = generator_trainer_fake.train_on_batch(x=[batch_real, batch_random[3]], y=np.ones(shape=(batch_size, 1)))

            batch_tqdm.set_postfix(dict(zip(['d_real', 'd_fake', 'g_real', 'g_fake'], ['{:.03f}'.format(loss) for loss in losses])))
        generator.save('{}_g.h5'.format(train_label), include_optimizer=False)
        predict(generator_path='{}_g.h5'.format(train_label), sample_count=3, save_dir_path=result_dir_path)


#
def predict(generator_path: str, sample_count: int, save_dir_path: str=None, show: bool=False):
    if not os.path.isfile(generator_path):
        raise FileNotFoundError(generator_path)
    if save_dir_path is not None:
        if os.path.isfile(save_dir_path):
            raise FileExistsError(save_dir_path)
        os.makedirs(save_dir_path, exist_ok=True)
    if sample_count <= 0:
        raise ValueError('sample_count should be a positive integer')

    #
    label_prefix = '.'.join(os.path.splitext(os.path.basename(generator_path))[:-1])
    label_index = int(time()) % 10000000

    #
    model: Model = load_model(generator_path, dict(leaky_relu=leaky_relu))
    model_input_shape = [int(item) for item in model.get_input_shape_at(0)[1:]]

    #
    predicted = model.predict(np.random.randn(sample_count, *model_input_shape))
    for i in range(sample_count):
        image = Image.fromarray(denormalized(predicted[i])).resize((128, 128), resample=Image.BICUBIC)
        if save_dir_path is not None:
            image.save(os.path.join(save_dir_path, '{}_{}_{:02}.png'.format(label_prefix, label_index, i)))
        if show:
            image.show()


#
def main():
    train_label = 'cifar'

    real_data = keras.datasets.cifar10.load_data()[0][0]
    generator = build_cifar10_generator(random_dims=64)
    discriminator = build_cifar10_discriminator(image_shape=(32, 32, 3))

    train(real_data=real_data,
          generator=generator,
          discriminator=discriminator,
          target_epochs=30,
          batch_size=64,
          train_label=train_label,
          random_dims=64,
          image_shape=real_data.shape[1:],
          result_dir_path='result')
    predict(generator_path='./{}_g.h5'.format(train_label), sample_count=5, save_dir_path='result')


if __name__ == '__main__':
    main()