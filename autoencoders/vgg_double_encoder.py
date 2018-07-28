import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.initializers import Normal


class VAE_encoder_double(chainer.Chain):
    """
    Implements an encoder that consists of 16 layers of pretrained VGG, followed by a 
    shared FCC layer. Then it bifurcates into two different VAE components, one for 
    regular images and the other for images that had been masked with attention.

    Intent: different encodings for the masked and unmasked images. The masked images
    have details only in the attention part, so we hope that they are reproduced better.
    The original images have all the screen so it keep obstacles etc.

    Validate: after training, check if the attention models are indeed reproducing better
    detail. We can actually try both of the paths on both kind of images. Do we need both
    of them???
    """
    def __init__(self, latent_size=16):
        super(VAE_encoder_double, self).__init__(
            # a pretrained VGG feature extractor, not trained in this network
            vgg = L.VGG16Layers(),
            # a shared, trainable FC layer, that will connect into the pool5 layer 
            # of the VGG. This is shared between the regular and attention path
            # Intent: ??? as the VGG is common, why not the first FC layer as well ???
            fc6 = L.Linear(512 * 7 * 7, 4096, initialW=Normal(0.02)),
            # a trainable VAE for the regular images
            fc7 = L.Linear(4096, 4096, initialW=Normal(0.02)),
            mean=L.Linear(4096, latent_size, initialW=Normal(0.02)),
            var=L.Linear(4096, latent_size, initialW=Normal(0.02)),
            # a trainable VAE for the attention masked images
            # Intent: ??? we have a different feature set for the attention masked 
            # images and the non-attention masked images
            fc7_att = L.Linear(4096, 4096, initialW=Normal(0.02)),
            mean_att = L.Linear(4096, latent_size, initialW=Normal(0.02)),
            var_att=L.Linear(4096, latent_size, initialW=Normal(0.02)),
        )
        self.vgg.disable_update() 

    def __call__(self, x, att=False, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            # starting from the last pooling layer of VGG16
            conv5_3 = self.vgg(x, layers=['pool5'])['pool5']
            h0 = F.leaky_relu(self.fc6(conv5_3))
            h0 = F.dropout(h0, ratio=0.5)
            # intent: regular images go through and train fc7,mean and var, while 
            # attention masked images, the other way around
            # FIXME make an array 0,1 for the attention models and not if
            if not att:
                # standard VAE encoder implementation
                h1 = F.leaky_relu(self.fc7(h0))
                h1 = F.dropout(h1, ratio=0.5)
                mean = self.mean(h1)
                var = F.tanh(self.var(h1))
                # the latent shape encoding z is a sample from the mean and var
                rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
                z = mean + F.exp(var) * Variable(rand)
                return z, mean, var
            else:
                h1 = F.leaky_relu(self.fc7_att(h0))
                h1 = F.dropout(h1, ratio=0.5)
                mean = self.mean_att(h1)
                var = F.tanh(self.var_att(h1))
                rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
                z = mean + F.exp(var) * Variable(rand)
                return z, mean, var