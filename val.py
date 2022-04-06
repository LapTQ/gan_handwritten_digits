import argparse
import utils.utils as my_utils
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def run(num_images):
    generator = keras.models.load_model('saved_model/generator')
    z_dim = generator.layers[0].input.shape[1]
    noise_batch = my_utils.get_noise(num_images, z_dim)
    fake_image_batch = generator(noise_batch)
    my_utils.plot_images(fake_image_batch)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument('--num-images', default=4, type=int)

    args = vars(ap.parse_args())

    run(args['num_images'])


