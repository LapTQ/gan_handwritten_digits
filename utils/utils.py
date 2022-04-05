import tensorflow as tf
import matplotlib.pyplot as plt

def get_noise(batch_size, z_dim, seed=42):
    return tf.random.normal(shape=(batch_size, z_dim), seed=seed)

def get_generator_loss(generator, discriminator, loss_object, noise_batch):
    generator.trainable = True
    fake_image_batch = generator(noise_batch)

    discriminator.trainable = False
    fake_image_preds = discriminator(fake_image_batch)
    desired_labels = tf.ones(fake_image_preds.shape)

    return loss_object(desired_labels, fake_image_preds)

def get_discriminator_loss(generator, discriminator, loss_object, noise_batch, real_image_batch):
    generator.trainable = False
    fake_image_batch = generator(noise_batch)
    fake_image_preds = discriminator(fake_image_batch)

    discriminator.trainable = True
    real_image_preds = discriminator(real_image_batch)

    fake_loss = loss_object(
        tf.zeros(fake_image_preds.shape),
        fake_image_preds
    )

    real_loss = loss_object(
        tf.ones(real_image_preds.shape),
        real_image_preds
    )

    return (fake_loss + real_loss) / 2.

def plot_images(images):
    """
    Arguments
    :param images: batch of images, shape (N, 28, 28)
    :return: None
    """
    collage = tf.concat([image for image in images], axis=1)
    plt.imshow(collage)

