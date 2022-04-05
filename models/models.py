import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Generator(keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.base_model = keras.Sequential(
            [
                layers.Dense(500),
                layers.LeakyReLU(0.2),
                layers.Dense(500),
                layers.LeakyReLU(0.2),
                layers.Dense(784, activation='sigmoid'),
                layers.Reshape((28, 28))
            ]
        )

    def call(self, z):
        """
        Arguments:
        :param z: a batch of noise vectors, shape (N, z_dim)
        :return: a batch of images, shape (N, 28, 28)
        """
        return self.base_model(z)


class Discriminator(keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.base_model = keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(500),
                layers.LeakyReLU(0.2),
                layers.Dense(200),
                layers.LeakyReLU(0.2),
                layers.Dense(1, activation='sigmoid')
            ]
        )

    def call(self, x):
        """
        Arguments
        :param x: batch of images, shape (N, 28, 28)
        :return: batch of real/fake prediction, shape (N, 1)
        """
        return self.base_model(x)






