import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.initializers import Normal

class Encoder(chainer.Chain):
    """
    Simple implementation of a convolutional encoder for VAE-GAN architecture
    
    This system takes as input only an image and not any condition.
    
    """
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        filter_size = 4
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Encoder, self).__init__(
            dc1=L.Convolution2D(channel, int(16 * density), filter_size, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(32 * density), filter_size, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(32 * density)),
            dc3=L.Convolution2D(int(32 * density), int(64 * density), filter_size, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), filter_size, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            mean=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                          initialW=Normal(0.02)),
            var=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                         initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h3 = F.leaky_relu(self.norm3(self.dc3(h2)))
            h4 = F.leaky_relu(self.norm4(self.dc4(h3)))
            mean = self.mean(h4)
            var = F.tanh(self.var(h4))
            rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            z = mean + F.exp(var) * Variable(rand)
            return z, mean, var


class Generator(chainer.Chain):
    """
    Simple implementation of a convolutional generator for VAE-GAN architecture. This ge
    
    This system takes as input only a vector and generates a 3-channel image.
    """
    def __init__(self, density=1, size=64, latent_size=100, channel=3):
        filter_size = 4
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Generator, self).__init__(
            g1=L.Linear(latent_size, initial_size * initial_size * int(128 * density),
                        initialW=Normal(0.02)),
            norm1=L.BatchNormalization(initial_size * initial_size * int(128 * density)),
            g2=L.Deconvolution2D(int(128 * density), int(64 * density), filter_size, stride=2, pad=1,
                                 initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(64 * density)),
            g3=L.Deconvolution2D(int(64 * density), int(32 * density), filter_size, stride=2, pad=1,
                                 initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(32 * density)),
            g4=L.Deconvolution2D(int(32 * density), int(16 * density), filter_size, stride=2, pad=1,
                                 initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(16 * density)),
            g5=L.Deconvolution2D(int(16 * density), channel, filter_size, stride=2, pad=1,
                                 initialW=Normal(0.02)),
        )
        self.density = density
        self.latent_size = latent_size
        self.initial_size = initial_size

    def __call__(self, z, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            h1 = F.reshape(F.relu(self.norm1(self.g1(z))),
                           (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
            h2 = F.relu(self.norm2(self.g2(h1)))
            h3 = F.relu(self.norm3(self.g3(h2)))
            h4 = F.relu(self.norm4(self.g4(h3)))
            return F.tanh(self.g5(h4))


class Discriminator(chainer.Chain):
    def __init__(self, density=1, size=64, channel=3):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Discriminator, self).__init__(
            dc1=L.Convolution2D(channel, int(16 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(32 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(32 * density)),
            dc3=L.Convolution2D(int(32 * density), int(64 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            dc5=L.Linear(initial_size * initial_size * int(128 * density), 2,
                         initialW=Normal(0.02)),
            dc6=L.Linear(initial_size * initial_size * int(128 * density), 2,
                         initialW=Normal(0.02)),
        )

    def __call__(self, x, att=True, train=True):
        with chainer.using_config('train', train):
            h1 = F.leaky_relu(self.dc1(x))
            h1 = F.dropout(h1, ratio=0.5)
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h2 = F.dropout(h2, ratio=0.5)
            h3 = F.leaky_relu(self.norm3(self.dc3(h2)))
            h3 = F.dropout(h3, ratio=0.5)
            h4 = F.leaky_relu(self.norm4(self.dc4(h3)))
            if att:
                return self.dc5(h4), h3
            else:
                return self.dc6(h4), h3