# Deep Convolutional Generative Adverserial Network

This project makes an attempt to implement a DCGAN network consisting of a generator n/w and discriminator n/w.
The dataset used to train the generator is CIFAR-10 because of the simplicity.

## Implementation Details

* python 3+
* Tensorflow 2.0 
* CIFAR - 10

## Architecture
### Generator

1. Dense 8x8x256 (Reshaped) + LeakyReLU
2. Upconvolution Conv2DTranspose (5x5x128-s-1) + BatchNorm + LeakyReLU
3. Upconvolution Conv2DTranspose (5x5x64-s-2) + BatchNorm + LeakyReLU
4. Upconvolution Conv2DTranspose (5x5x3-s-2) + BatchNorm + LeakyReLU

### Discriminator

1. Conv2D (5x5x64-s-2) + BatchNorm + LeakyReLU + Dropout
2. Conv2D (5x5x128-s-2) + BatchNorm + LeakyReLU
3. Conv2D (5x5x128-s-2) + BatchNorm + LeakyReLU
4. Dense (1) + Sigmoid

