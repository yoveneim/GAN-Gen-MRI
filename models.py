"""Generator and discriminator used by 4.GAN.ipynb and 5.generation_using_model.ipynb."""

import tensorflow as tf
from tensorflow.keras import layers

NOISE_DIM = 512
IMG_SIZE = 256


def generator_model():
    # 512-d noise vector -> 256x256x1 image in [-1, 1]
    model = tf.keras.Sequential()
    model.add(layers.Reshape(target_shape=[1, 1, NOISE_DIM], input_shape=[NOISE_DIM]))
    assert model.output_shape == (None, 1, 1, 512)

    model.add(layers.Conv2DTranspose(filters=256, kernel_size=4))
    model.add(layers.Activation('relu'))
    assert model.output_shape == (None, 4, 4, 256)

    # Each block: conv -> batchnorm -> relu -> upsample (x2)
    for filters, kernel_size, size in [(256, 4, 8), (128, 4, 16), (64, 3, 32), (32, 3, 64), (16, 3, 128)]:
        model.add(layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same'))
        model.add(layers.BatchNormalization(momentum=0.7))
        model.add(layers.Activation('relu'))
        model.add(layers.UpSampling2D())
        assert model.output_shape == (None, size, size, filters)

    model.add(layers.Conv2D(filters=8, kernel_size=3, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.UpSampling2D())
    assert model.output_shape == (None, 256, 256, 8)

    model.add(layers.Conv2D(filters=1, kernel_size=3, padding='same'))
    model.add(layers.Activation('tanh'))
    assert model.output_shape == (None, IMG_SIZE, IMG_SIZE, 1)

    return model


def discriminator_model():
    # 256x256x1 image -> probability that the image is real
    model = tf.keras.Sequential()

    # Add Gaussian noise to prevent discriminator overfitting
    model.add(layers.GaussianNoise(0.2, input_shape=[IMG_SIZE, IMG_SIZE, 1]))

    # 256x256x1 -> 128x128x8 (no batchnorm on the first block)
    model.add(layers.Conv2D(filters=8, kernel_size=3, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.AveragePooling2D())

    # 128x128x8 -> ... -> 4x4x256
    for filters in [16, 32, 64, 128, 256]:
        model.add(layers.Conv2D(filters=filters, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization(momentum=0.7))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.AveragePooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model
