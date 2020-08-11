import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from model import generator,discriminator, DCGAN
import os
import pickle

IMAGE_H, IMAGE_W = 32, 32
CLASS            = 10
BATCH_SIZE       = 32
IMAGE_COUNT      = 60000
EPOCHS           = 100

MODEL_SAVE_DIR   = './model/'
MODEL_NAME       = 'DCGAN'
MODEL_RESULT     = "%s-results.pickle" % MODEL_NAME

model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model_ckpt_path = os.path.join(model_save_path, "model-ckpt")
model_rslt_path = os.path.join(model_save_path, MODEL_RESULT)

LATENT_DIMS = 1000
GEN_SPECS = {
                    'layer_1' : {
                        'layer_type' : 'Dense',
                        'kernel_size' : 8,
                        'filters': 256  # 8x8x256
                    },
                    'layer_2': {
                        'layer_type': 'Conv2DTranspose',
                        'kernel_size': 5,
                        'filters': 128,
                        'stride' : 1  # 8x8x128
                    },
                    'layer_3': {
                        'layer_type': 'Conv2DTranspose',
                        'kernel_size': 5,
                        'filters': 64,
                        'stride': 2  # 16x16x64
                    },
                    'layer_4': {
                        'layer_type': 'Conv2DTranspose',
                        'kernel_size': 5,
                        'filters': 3,
                        'stride': 2  # 32x32x3
                    }
                }

DISC_SPECS = {
                    'layer_1' : { # 32x32x3
                        'layer_type' : 'Conv2D',
                        'kernel_size' : 5,
                        'filters': 64,
                        'stride' : 2  # 16x16x64
                    },
                    'layer_2': {
                        'layer_type': 'Conv2D',
                        'kernel_size': 5,
                        'filters': 128,
                        'stride' : 2 # 8x8x128
                    },
                    'layer_3': {
                        'layer_type': 'Conv2D',
                        'kernel_size': 5,
                        'filters': 128,
                        'stride': 2 # 4x4x128
                    },
                    'layer_4': {
                        'layer_type': 'Dense'
                    } # 1
                }
def feature_normalize(features):
    return (features/255 - 0.5) / 0.5

def feature_denormalize(features):
    return (features + 1) / 2

print('Testing generator output sample.....')
my_gen = generator(latent_dim = (1000,), specs = GEN_SPECS, name = 'gen_1')
noise = tf.random.normal([1, 1000])
generated_image = my_gen.model(noise, training=False)
print(generated_image[0, ...].shape)
plt.imshow(generated_image[0, ...]/np.max(generated_image[0, ...]))
plt.title('Sample generator output')
plt.show()

print('Testing discriminator output on generated sample.....')
my_disc = discriminator(specs = DISC_SPECS, image_w = 32, image_h = 32, name = 'disc_1')
label = my_disc.model(generated_image)
print('Predicted label:', label)

print('Loading CIFAR-10 dataset.....')
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#print('Training image shape:', train_images.shape)
#print('Test image shape:', test_images.shape)

data_image = np.concatenate((train_images, test_images), axis = 0)
data_labels = np.concatenate((train_labels, test_labels), axis = 0)
#print('Total image shape:', data_image.shape)
dataset = tf.data.Dataset.from_tensor_slices(data_image)
dataset = dataset.repeat().shuffle(1024).batch(BATCH_SIZE)

model = DCGAN(latent_dim = (LATENT_DIMS,), gen_specs = GEN_SPECS, disc_specs = DISC_SPECS,
              image_h =IMAGE_H,image_w = IMAGE_W)

generator_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

@tf.function
def train_step(x, z):
    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        generator_loss = model.generator_loss(z)
        discriminator_loss = model.discriminator_loss(x, z)

        grads_generator_loss = generator_tape.gradient(
            target=generator_loss, sources=model.generator.trainable_variables
        )
        grads_discriminator_loss = discriminator_tape.gradient(
            target=discriminator_loss, sources=model.discriminator.trainable_variables
        )

        discriminator_opt.apply_gradients(
            zip(grads_discriminator_loss, model.discriminator.trainable_variables)
        )
        generator_opt.apply_gradients(
            zip(grads_generator_loss, model.generator.trainable_variables)
        )

    return generator_loss, discriminator_loss

ckpt = tf.train.Checkpoint(generator=model.generator, discriminator=model.discriminator)

steps_per_epoch = IMAGE_COUNT // BATCH_SIZE
train_steps = steps_per_epoch * EPOCHS
print('GAN Training....')
generator_losses = []
discriminator_losses = []
generator_losses_epoch = []
discriminator_losses_epoch = []
x_fakes = []
for i in range(1, train_steps + 1):
    epoch = i // steps_per_epoch
    #print('Inside for loop.....')
    print("Epoch: %i ====> %i / %i" % (epoch + 1, i % steps_per_epoch, steps_per_epoch))

    for x in dataset.take(1):
        x_i = feature_normalize(x)
        z_i = np.random.normal(size=[BATCH_SIZE, LATENT_DIMS]).astype(np.float32)

        generator_loss_i, discriminator_loss_i = train_step(x_i, z_i)

        generator_losses.append(generator_loss_i)
        discriminator_losses.append(discriminator_loss_i)

    if i % steps_per_epoch == 0:
        x_fake = model.generator(z_i, training=False)
        x_fake = feature_denormalize(x_fake)

        generator_loss_epoch = np.mean(generator_losses[-steps_per_epoch:])
        discriminator_loss_epoch = np.mean(discriminator_losses[-steps_per_epoch:])

        print("Epoch: %i,  Generator Loss: %f,  Discriminator Loss: %f" % \
              (epoch, generator_loss_epoch, discriminator_loss_epoch)
              )

        generator_losses_epoch.append(generator_loss_epoch)
        discriminator_losses_epoch.append(discriminator_loss_epoch)

        x_fakes.append(x_fake)

        ckpt.save(file_prefix="model_ckpt_")

        with open(model_rslt_path, "wb") as f:
            pickle.dump((generator_losses_epoch, discriminator_losses_epoch, x_fakes), f)