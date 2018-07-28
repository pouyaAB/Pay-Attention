import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.initializers import Normal

class Encoder_text_tower(chainer.Chain):
    """
    An implementation of the "Tower" model of the VAE encoder from the paper 
    'Neural scene representation and rendering, by S. M. Ali Eslami and others at DeepMind.
    The exact numbers of the layers and were changed.

    It is basically a cVAE with multi-dimensional conditions.

    v - human level properties HLP, human classification ???? find a good name

    This system takes as input an image and two one-hot vectors corresponding to factorized
    features of the main object encoded in the image. For instance, if the image contains a 
    red sphere, the inputs will <image>,"red","round". 

    Intent: The hypothesis is that by providing HLPs during training and also during testing,
    we get a better encoding _of the particular object_. 

    Validation: We can check the reconstruction error metric, but we can also check this 
    with visual inspection of the reconstructed version for different HLP inputs
    """
    def __init__(self, density=1, size=64, latent_size=100, channel=3, hidden_dim=100, num_objects=10, num_describtions=10):
        """
        density - a scaling factor for the number of channels in the convolutional layers. It is multiplied by at least
        16,32,64 and 128 as we go deeper. 
        Use: using density=8 when training the VAE separately. using density=4 when training end to end
        Intent: increase the number of features in the convolutional layers. 
        """
        assert (size % 16 == 0)
        second_size = size / 4
        initial_size = size / 16
        super(Encoder_text_tower, self).__init__(
            # 
            dc1=L.Convolution2D(channel, int(16 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
                                # can we write comments here
            dc2=L.Convolution2D(int(16 * density), int(32 * density), 3, stride=2, pad=1,

                                initialW=Normal(0.02)),
            # extra layers added to make it deeper with stride = 1
            dc1_=L.Convolution2D(int(16 * density), int(16 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            dc2_=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(32 * density)),
            norm2_=L.BatchNormalization(int(32 * density)),
            # "plus layer" another extra layer added to make it deeper with stride = 1 but this one has 
            # a skip connection between input and output
            dc2_p=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2_p=L.BatchNormalization(int(32 * density)),
            dc3=L.Convolution2D(int(32 * density + 7), int(64 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            dc3_=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_=L.BatchNormalization(int(64 * density)),
            dc3_p=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_p=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),

            toConv=L.Linear(num_objects + num_describtions, second_size * second_size * 7, initialW=Normal(0.02)),
            mean=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                          initialW=Normal(0.02)),
            var=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                         initialW=Normal(0.02)),
        )

    def __call__(self, x, objects_one_hot, descs_one_hot, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            h0 = self.toConv(F.concat((objects_one_hot, descs_one_hot), axis=-1))
            h0 = F.reshape(h0, (h0.shape[0], 7, 32, 32))
            h1 = F.leaky_relu(self.dc1(x))
            h1_ = F.leaky_relu(self.dc1_(h1))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1_)))
            h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2)))
            h2_p = F.leaky_relu(self.norm2_p(self.dc2_p(h2_)))
            # skip connection
            h2_ = h2_ + h2_p
            # this is the point where the two one-hot encodings converted to a conv mode 
            # are inserted into the chain of features
            h2_ = F.concat((h2_, h0), axis=1)
            h3 = F.leaky_relu(self.norm3(self.dc3(h2_)))
            h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3)))
            h3_p = F.leaky_relu(self.norm3_p(self.dc3_p(h3)))
            # skip connection
            h3_ = h3_ + h3_p
            h4 = F.leaky_relu(self.norm4(self.dc4(h3_)))
            # from here, this is the regular VAE sampling
            mean = self.mean(h4)
            var = F.tanh(self.var(h4))
            rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            z = mean + F.exp(var) * Variable(rand)
            return z, mean, var


class Generator_text(chainer.Chain):
    """
    This implemention is very similar to the encoder_text_tower. 
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
        filter_size = 2
        intermediate_size = size / 8
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Generator_text, self).__init__(
            g0=L.Linear(num_objects + num_describtions, intermediate_size * intermediate_size * 7, initialW=Normal(0.02)),
            g1=L.Linear(latent_size, initial_size * initial_size * int(128 * density),
                        initialW=Normal(0.02)),
            norm1=L.BatchNormalization(initial_size * initial_size * int(128 * density)),
            g2=L.Deconvolution2D(int(128 * density), int(64 * density), filter_size, stride=2, pad=0,
                                 initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(64 * density)),
            # An extra layer to make the network deeper and not changing the feature sizes
            g2_=L.Deconvolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            norm2_=L.BatchNormalization(int(64 * density)),
            # "plus layer" another extra layer added to make it deeper with stride = 1 but this one has 
            # a skip connection between input and output
            g2_p=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2_p=L.BatchNormalization(int(64 * density)),
            g3=L.Deconvolution2D(int(64 * density + 7), int(32 * density), filter_size, stride=2, pad=0,
                                 initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(32 * density)),
            g3_=L.Deconvolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            norm3_=L.BatchNormalization(int(32 * density)),
            g3_p=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_p=L.BatchNormalization(int(32 * density)),
            g4=L.Deconvolution2D(int(32 * density), int(16 * density), filter_size, stride=2, pad=0,
                                 initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(16 * density)),
            g4_=L.Deconvolution2D(int(16 * density), int(16 * density), 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            norm4_=L.BatchNormalization(int(16 * density)),
            g5=L.Deconvolution2D(int(16 * density), channel, filter_size, stride=2, pad=0,
                                 initialW=Normal(0.02)),
        )
        self.density = density
        self.latent_size = latent_size
        self.initial_size = initial_size

    def __call__(self, z, objs, descs, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            h0 = self.g0(F.concat((objs, descs), axis=-1))
            h0 = F.reshape(h0, (h0.shape[0], 7, 16, 16))
            h1 = F.reshape(F.relu(self.norm1(self.g1(z))),
                           (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
            h2 = F.relu(self.norm2(self.g2(h1)))
            h2_ = F.relu(self.norm2_(self.g2_(h2)))
            h2_p = F.relu(self.norm2_p(self.g2_p(h2_)))
            # skip connection
            h2_ = h2_ + h2_p
            h2_ = F.concat((h2_, h0), axis=1)
            h3 = F.relu(self.norm3(self.g3(h2_)))
            h3_ = F.relu(self.norm3_(self.g3_(h3)))
            h3_p = F.relu(self.norm3_p(self.g3_p(h3_)))
            # skip connection
            h3_ = h3_ + h3_p
            h4 = F.relu(self.norm4(self.g4(h3_)))
            h4_ = F.relu(self.norm4_(self.g4_(h4)))
            return F.tanh(self.g5(h4_))

class Discriminator_cond(chainer.Chain):
    """
    A discriminator for classifying the both the regular and masked images into their HLP groups.
    The discrimiantor can recieve both regular and masked images as input and it will try to 
    classify them based on the object of interest in the image. The discriminator will either 
    mark the image as fake or it will match it to one the shapes and colors. 

    The regular and masked images have will go through different convolution and FF layers at
    the end. The first 7 convoltion layers are shared and there is two additional convolution 
    layers for each regular and masked images.

    The disciminator will be used in an adversarial setup witn the encoder and the generator.
    The disciminator tries to mark the iamges generated by the generator as fake and classify 
    real images to their correct class.

    Validation: One can check the classification error for real and fake images
    """
    def __init__(self, density=1, size=64, channel=3, num_objects=10, num_describtions=10):
        assert (size % 16 == 0)
        second_size = size / 4
        initial_size = size / 16
        super(Discriminator_cond, self).__init__(
            # shared layers between regualr and masked images
            dc1=L.Convolution2D(channel, int(16 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(32 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc1_=L.Convolution2D(int(16 * density), int(16 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            # extra layers added to make it deeper with stride = 1
            dc2_=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(32 * density)),
            norm2_=L.BatchNormalization(int(32 * density)),
            # "plus layer" another extra layer added to make it deeper with stride = 1 but this one has 
            # a skip connection between input and output
            dc2_p=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2_p=L.BatchNormalization(int(32 * density)),
            dc3=L.Convolution2D(int(32 * density + 7), int(64 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            dc3_=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_=L.BatchNormalization(int(64 * density)),

            #seperated layers for the regular images
            dc3_p=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_p=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),

            # seperated layers for the masked images
            dc3_p_att=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_p_att=L.BatchNormalization(int(64 * density)),
            dc4_att=L.Convolution2D(int(64 * density), int(128 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4_att=L.BatchNormalization(int(128 * density)),

            toConv=L.Linear(num_objects + num_describtions, second_size * second_size * 7, initialW=Normal(0.02)),
            FC1=L.Linear(initial_size * initial_size * int(128 * density), 2, initialW=Normal(0.02)),
            FC2=L.Linear(initial_size * initial_size * int(128 * density), 2, initialW=Normal(0.02))
        )

    def __call__(self, x, objects_one_hot, descs_one_hot, att=True, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            # h0 = F.concat((x, objects, descs), axis=1)
            h0 = self.toConv(F.concat((objects_one_hot, descs_one_hot), axis=-1))
            h0 = F.reshape(h0, (h0.shape[0], 7, 32, 32))
            h1 = F.leaky_relu(self.dc1(x))
            h1_ = F.leaky_relu(self.dc1_(h1))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1_)))
            h2_ = F.leaky_relu(self.norm2_(self.dc2_(h2)))
            h2_p = F.leaky_relu(self.norm2_p(self.dc2_p(h2_)))
            h2_ = h2_ + h2_p
            h2_ = F.concat((h2_, h0), axis=1)
            h3 = F.leaky_relu(self.norm3(self.dc3(h2_)))
            h3_ = F.leaky_relu(self.norm3_(self.dc3_(h3)))
            if att:
                h3_p = F.leaky_relu(self.norm3_p_att(self.dc3_p_att(h3)))
                h3_ = h3_ + h3_p
                h4 = F.leaky_relu(self.norm4_att(self.dc4_att(h3_)))
                return self.FC1(h4), h3_p
            else:
                h3_p = F.leaky_relu(self.norm3_p(self.dc3_p(h3)))
                h3_ = h3_ + h3_p
                h4 = F.leaky_relu(self.norm4(self.dc4(h3_)))
                return self.FC2(h4), h3_p