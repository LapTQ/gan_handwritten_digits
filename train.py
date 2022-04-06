import tensorflow as tf
from tensorflow import keras
from models.models import Generator, Discriminator
import matplotlib.pyplot as plt
import utils.utils as my_utils
import argparse
from tqdm import tqdm
import os
import shutil

def run(size_in_batch,
        batch_size,
        epochs,
        z_dim,
        learning_rate,
        display_step,
        run_eagerly):

    tf.config.run_functions_eagerly(run_eagerly)

    (train_images, _), (__, ___) = keras.datasets.mnist.load_data()

    train_images = train_images[:size_in_batch * batch_size]

    train_images = tf.reshape(train_images, (-1, batch_size, *train_images.shape[1:]))

    generator = Generator()
    discriminator = Discriminator()

    loss_object = keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):

        print(f'EPOCH [{epoch + 1}/{epochs}]')

        for image_batch in tqdm(train_images):
            noise_batch = my_utils.get_noise(batch_size, z_dim)

            with tf.GradientTape() as tape:
                discriminator_loss = my_utils.get_discriminator_loss(generator, discriminator, loss_object, noise_batch, image_batch)
            gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
            optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

        for _ in range(6 * len(train_images)):
            noise_batch = my_utils.get_noise(batch_size, z_dim)

            with tf.GradientTape() as tape:
                generator_loss = my_utils.get_generator_loss(generator, discriminator, loss_object, noise_batch)
            gradients = tape.gradient(generator_loss, generator.trainable_variables)
            optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

        if epoch % display_step == 0:
            noise_batch = my_utils.get_noise(5, z_dim)
            fake_image_batch = generator(noise_batch)
            my_utils.plot_images(fake_image_batch)
            plt.savefig(os.path.join('plots', ('000000' + str(epoch + 1) + '.jpg')[-9:]))


    generator.save('saved_model/generator')
    discriminator.save('saved_model/discriminator')


if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument('--size-in-batch', default=100, type=int)
    ap.add_argument('--epochs', required=True, type=int)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--z_dim', type=int, default=512)
    ap.add_argument('--learning-rate', type=float, default=0.01)
    ap.add_argument('--display-step', type=int, default=5)
    ap.add_argument('--run-eagerly', type=bool, default=False)

    args = vars(ap.parse_args())

    shutil.rmtree('saved_model', ignore_errors=True)
    os.system('mkdir saved_model')
    shutil.rmtree('plots', ignore_errors=True)
    os.system('mkdir plots')

    run(args['size_in_batch'],
        args['batch_size'],
        args['epochs'],
        args['z_dim'],
        args['learning_rate'],
        args['display_step'],
        args['run_eagerly'])







