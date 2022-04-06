import tensorflow as tf
from tensorflow import keras
from models.models import Generator, Discriminator
import matplotlib.pyplot as plt
import utils.utils as my_utils
import argparse
from tqdm import tqdm

def run(size_in_batch, batch_size, epochs, z_dim, display_step):
    (train_images, _), (__, ___) = keras.datasets.mnist.load_data()

    train_images = train_images[:size_in_batch * batch_size]

    train_images = tf.reshape(train_images, (-1, batch_size, *train_images.shape[1:]))

    generator = Generator()
    discriminator = Discriminator()

    loss_object = keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(0.001)

    for epoch in range(epochs):

        for image_batch in tqdm(train_images):
            noise_batch = my_utils.get_noise(batch_size, z_dim)

            with tf.GradientTape() as tape:
                discriminator_loss = my_utils.get_discriminator_loss(generator, discriminator, loss_object, noise_batch, image_batch)
            gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
            optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

            for _ in range(6):
                with tf.GradientTape() as tape:
                    generator_loss = my_utils.get_generator_loss(generator, discriminator, loss_object, noise_batch)
                gradients = tape.gradient(generator_loss, generator.trainable_variables)
                optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

        if epoch % display_step == 0:
            my_utils.plot_images(train_images[0][:5])
            plt.show()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--size-in-batch', required=True, default=100, type=int)
    ap.add_argument('--epochs', required=True, type=int)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--z_dim', type=int, default=512)
    ap.add_argument('--display-step', type=int, default=5)

    args = vars(ap.parse_args())

    run(args['size_in_batch'], args['batch_size'], args['epochs'], args['z_dim'], args['display_step'])







