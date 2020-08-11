import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose,\
    Flatten, LeakyReLU, BatchNormalization, Dropout
import matplotlib.pyplot as plt
class generator(Model):

    def __init__(self, latent_dim, specs : dict, name, **kwargs):
        super(generator, self).__init__(name = name, **kwargs)

        # generator model definition
        model = None
        # input layer
        latent_input = Input(shape = latent_dim, dtype='float32')

        #layer 1
        name = specs['layer_1']['layer_type']
        kernel = specs['layer_1']['kernel_size']
        filter_count = specs['layer_1']['filters']

        x = Dense( units = kernel*kernel*filter_count,
                   input_shape = latent_dim, use_bias = False,
                   name = 'layer_1_' + name)(latent_input)

        x = LeakyReLU(alpha=0.1)(x)
        x = Reshape((kernel, kernel, filter_count))(x)
        #print(x._shape)
        #print((None, kernel, kernel, filter_count))
        assert np.prod(x.shape[1:]) == (kernel * kernel * filter_count)

        # layer 2
        name = specs['layer_2']['layer_type']
        kernel = specs['layer_2']['kernel_size']
        filter_count = specs['layer_2']['filters']
        stride = specs['layer_2']['stride']

        x = Conv2DTranspose(kernel_size = (kernel, kernel),
                            filters = filter_count,
                            strides = (stride, stride),
                            padding = 'same',
                            use_bias=False,
                            name='layer_2_' + name)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        #assert x.output_shape == (None, kernel, kernel, filter_count)

        # layer 3
        name = specs['layer_3']['layer_type']
        kernel = specs['layer_3']['kernel_size']
        filter_count = specs['layer_3']['filters']
        stride = specs['layer_3']['stride']

        x = Conv2DTranspose(kernel_size=(kernel, kernel),
                            filters=filter_count,
                            strides=(stride, stride),
                            padding='same',
                            use_bias=False,
                            name='layer_3_' + name)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # layer 4
        name = specs['layer_4']['layer_type']
        kernel = specs['layer_4']['kernel_size']
        filter_count = specs['layer_4']['filters']
        stride = specs['layer_4']['stride']

        x = Conv2DTranspose(kernel_size=(kernel, kernel),
                            filters=filter_count,
                            strides=(stride, stride),
                            padding='same',
                            use_bias=False,
                            name='layer_4_' + name, activation='tanh')(x)
        generator_image = x
        assert np.prod(x.shape[1:]) == (32 * 32 * 3)

        self.model = Model(latent_input, generator_image)


class discriminator(Model):

    def __init__(self, specs : dict, name, image_w, image_h, **kwargs):
        super(discriminator, self).__init__(name = name, **kwargs)

        # image parameters
        self.image_w = image_w
        self.image_h = image_h

        # discriminator model definition
        model = None
        # input layer
        input = Input(shape = (self.image_h, self.image_w, 3), dtype='float32')

        #layer 1
        name = specs['layer_1']['layer_type']
        kernel = specs['layer_1']['kernel_size']
        filter_count = specs['layer_1']['filters']
        stride = specs['layer_1']['stride']

        x = Conv2D(kernel_size = (kernel, kernel),
                            filters = filter_count,
                            strides = (stride, stride),
                            padding = 'same',
                            use_bias=False,
                            name = 'layer_1_' + name)(input)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.3)(x)

        # layer 2
        name = specs['layer_2']['layer_type']
        kernel = specs['layer_2']['kernel_size']
        filter_count = specs['layer_2']['filters']
        stride = specs['layer_2']['stride']

        x = Conv2D(kernel_size = (kernel, kernel),
                            filters = filter_count,
                            strides = (stride, stride),
                            padding = 'same',
                            use_bias=False,
                            name='layer_2_' + name)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.3)(x)


        # layer 3
        name = specs['layer_3']['layer_type']
        kernel = specs['layer_3']['kernel_size']
        filter_count = specs['layer_3']['filters']
        stride = specs['layer_3']['stride']

        x = Conv2D(kernel_size=(kernel, kernel),
                            filters=filter_count,
                            strides=(stride, stride),
                            padding='same',
                            use_bias=False,
                            name='layer_3_' + name)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.3)(x)

        # layer 4
        name = specs['layer_4']['layer_type']
        x = Flatten()(x)
        x = Dense(units=1, activation = 'sigmoid', name= 'layer_4' + name)(x)
        label = x

        self.model = Model(input, label)


class DCGAN(object):
    def __init__(
            self,
            latent_dim,
            gen_specs,
            disc_specs,
            image_w,
            image_h
    ):
        self.latent_dim = latent_dim
        self.gen_specs = gen_specs
        self.disc_specs = disc_specs
        self.generator = generator(self.latent_dim, self.gen_specs, name = 'my_gen').model
        self.discriminator = discriminator(self.disc_specs, name = 'my_disc', image_w = image_w, image_h = image_h).model

    def generator_loss(self, z):
        x_fake = self.generator(z, training=True)
        fake_score = self.discriminator(x_fake, training=True)

        loss = tf.keras.losses.binary_crossentropy(
            y_true=tf.ones_like(fake_score), y_pred=fake_score, from_logits=False
        )

        return loss

    def discriminator_loss(self, x, z):
        x_fake = self.generator(z, training=True)
        fake_score = self.discriminator(x_fake, training=True)
        true_score = self.discriminator(x, training=True)

        loss = \
            tf.keras.losses.binary_crossentropy(
                y_true=tf.ones_like(true_score), y_pred=true_score, from_logits=False
            ) + \
            tf.keras.losses.binary_crossentropy(
                y_true=tf.zeros_like(fake_score), y_pred=fake_score, from_logits=False
            )

        return loss

gen_specs = {
                    'layer_1' : {
                        'layer_type' : 'Dense',
                        'kernel_size' : 8,
                        'filters': 256 # 8x8x256
                    },
                    'layer_2': {
                        'layer_type': 'Conv2DTranspose',
                        'kernel_size': 5,
                        'filters': 128,
                        'stride' : 1 # 8x8x128
                    },
                    'layer_3': {
                        'layer_type': 'Conv2DTranspose',
                        'kernel_size': 5,
                        'filters': 64,
                        'stride': 2 # 16x16x64
                    },
                    'layer_4': {
                        'layer_type': 'Conv2DTranspose',
                        'kernel_size': 5,
                        'filters': 3,
                        'stride': 2 # 32x32x3
                    }
                }
