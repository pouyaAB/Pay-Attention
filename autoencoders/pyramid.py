import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.initializers import Normal

class Encoder_text(chainer.Chain):
    """
    An implementation of the "pyramid" model of the VAE encoder from the paper 
    'Neural scene representation and rendering, by S. M. Ali Eslami and others at DeepMind.
    The exact numbers of the layers and were changed.

    It is basically a cVAE with multi-dimensional conditions.

    This system takes as input an image and two floting point numbers corresponding to factorized
    features of the main object encoded in the image. For instance, if the image contains a 
    red sphere, the inputs will <image>,"red","round". 

    Intent: The hypothesis is that by providing HLPs during training and also during testing,
    we get a better encoding _of the particular object_. 

    Validation: We can check the reconstruction error metric, but we can also check this 
    with visual inspection of the reconstructed version for different HLP inputs
    """
    def __init__(self, density=1, size=64, latent_size=100, channel=3, hidden_dim=100):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Encoder_text, self).__init__(
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
            mean=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                          initialW=Normal(0.02)),
            var=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                         initialW=Normal(0.02)),
        )

    def __call__(self, x, objects, descs, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            h0 = F.concat((x, objects, descs), axis=1)
            h1 = F.leaky_relu(self.dc1(h0))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h3 = F.leaky_relu(self.norm3(self.dc3(h2)))
            h4 = F.leaky_relu(self.norm4(self.dc4(h3)))
            mean = self.mean(h4)
            var = F.tanh(self.var(h4))
            rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            z = mean + F.exp(var) * Variable(rand)
            return z, mean, var

class Generator_text(chainer.Chain):
    """
    This implemention is very similar to the encoder_text. 
    Convolution layers has been replaced with deconvolution layers.

    This implemention recieves a latent vector plus two and two one-hot vectors corresponding to factorized
    features of the main object encoded in the image. For instance, if the image contains a 
    red sphere, the inputs will <image>,"red","round". 

    Intent: The hypothesis is that by providing HLPs during training and also during testing,
    we get a better generative results _of the particular object_. 

    Validation: We can check the reconstruction error metric, but we can also check this 
    with visual inspection of the reconstructed version for different HLP inputs
    """
    def __init__(self, density=1, size=64, latent_size=100, channel=3, num_objects=10, num_describtions=10):
        filter_size = 4
        # intermediate_size = 256
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Generator_text, self).__init__(
            # g0=L.Linear(latent_size + num_objects + num_describtions, intermediate_size, initialW=Normal(0.02)),
            g1=L.Linear(latent_size + num_objects + num_describtions, initial_size * initial_size * int(128 * density),
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

    def __call__(self, z, objects, descs, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            # h0 = self.g0(F.concat((z, objects, descs), axis=-1))
            h1 = F.reshape(F.relu(self.norm1(self.g1(F.concat((z, objects, descs), axis=-1)))),
                           (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
            h2 = F.relu(self.norm2(self.g2(h1)))
            h3 = F.relu(self.norm3(self.g3(h2)))
            h4 = F.relu(self.norm4(self.g4(h3)))
            return F.tanh(self.g5(h4))


class Discriminator_texual(chainer.Chain):
    """
    A simpler version of the discriminator with 6 shared convolution layers 
    and separate fully-connected layers for regular and masked images.

    
    """
    def __init__(self, density=1, size=64, channel=3, num_words=32, num_objects=10, num_describtions=10):
        assert (size % 16 == 0)
        self.num_objects = num_objects
        self.num_describtions = num_describtions
        initial_size = size / 16
        super(Discriminator_texual, self).__init__(
            dc1=L.Convolution2D(channel, int(16 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(32 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(32 * density)),
            # "plus layer" another extra layer added to make it deeper with stride = 1 but this one has 
            # a skip connection between input and output
            dc2_p=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2_p=L.BatchNormalization(int(32 * density)),
            dc3=L.Convolution2D(int(32 * density), int(64 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            # "plus layer" another extra layer added to make it deeper with stride = 1 but this one has 
            # a skip connection between input and output
            dc3_p=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_p=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            dc5=L.Linear(initial_size * initial_size * int(128 * density), num_objects, initialW=Normal(0.02)),
            dc6=L.Linear(initial_size * initial_size * int(128 * density), num_describtions, initialW=Normal(0.02)),
            dc8=L.Linear(initial_size * initial_size * int(128 * density), num_objects, initialW=Normal(0.02)),
            dc9=L.Linear(initial_size * initial_size * int(128 * density), num_describtions, initialW=Normal(0.02)),
        )

    def __call__(self, x, att=True, train=True):
        with chainer.using_config('train', train):
            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h2_p = F.leaky_relu(self.norm2_p(self.dc2_p(h2)))
            h2 = h2 + h2_p
            h3 = F.leaky_relu(self.norm3(self.dc3(h2)))
            h3_p = F.leaky_relu(self.norm3_p(self.dc3_p(h3)))
            h3 = h3 + h3_p
            h4 = F.leaky_relu(self.norm4(self.dc4(h3)))
            if att:
                return self.dc5(h4), self.dc6(h4), h3
            else:
                return self.dc8(h4), self.dc9(h4), h3