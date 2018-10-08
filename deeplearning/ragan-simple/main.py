from models import build_mnist_generator
from models import build_discriminator
from models import build_compiled_ragan_trainers
from models import leaky_relu

from keras.datasets.mnist import load_data

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


def normalized_to_image_array(x: np.ndarray):
    return np.squeeze(denormalized(x), axis=-1)


def image_array_to_normalized(x: np.ndarray):
    return np.expand_dims(normalized(x), axis=-1)


#
def train(target_epochs, batch_size: int, train_label: str, random_dims: int, image_shape: Collection[int], result_dir_path: str='result', load_epoch=None):
    if load_epoch is not None:
        raise NotImplementedError('load_epoch is not supported yet')
    if batch_size <= 0:
        raise ValueError('batch_size should be a positive integer')
    if random_dims <= 0:
        raise ValueError('random_dims should be a positive integer')
    if len(image_shape) != 3:
        raise ValueError('image_shape is not a shape with size and channels')

    #
    generator = build_mnist_generator(random_dims=random_dims)
    discriminator = build_discriminator(image_shape=image_shape)
    generator.summary()
    discriminator.summary()

    discriminator_trainer_real, discriminator_trainer_fake, generator_trainer_real, generator_trainer_fake = build_compiled_ragan_trainers(generator=generator,
                                                                                                                                           discriminator=discriminator,
                                                                                                                                           batch_size=batch_size,
                                                                                                                                           use_rmsprop=True)

    #
    (x_train, _), (_, _) = load_data()

    #
    epoch_tqdm =tqdm(range(target_epochs), desc='epochs', position=0)
    for _ in epoch_tqdm:
        losses = [0.] * 4
        np.random.shuffle(x_train)
        batch_tqdm = tqdm(range(len(x_train) // batch_size), desc='batches', position=1)
        for batch_index in batch_tqdm:
            batch_real = image_array_to_normalized(x_train[batch_index * batch_size:(batch_index + 1) * batch_size])
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
        image = Image.fromarray(normalized_to_image_array(predicted[i])).resize((28 * 4, 28 * 4))
        if save_dir_path is not None:
            image.save(os.path.join(save_dir_path, '{}_{}_{:02}.png'.format(label_prefix, label_index, i)))
        if show:
            image.show()


#
def main():
    train_label = 'mnist'

    train(target_epochs=30,
          batch_size=64,
          train_label=train_label,
          random_dims=64,
          image_shape=(28, 28, 1),
          result_dir_path='result')
    predict('./{}_g.h5'.format(train_label), 5, 'result')


if __name__ == '__main__':
    main()