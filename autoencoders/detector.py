import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.initializers import Normal


class Detector(chainer.Chain):
    """
    Based an idea written in 'Conditional Autoencoders with Adversarial Information Factorization' 
    by Antonia Creswell and others. 

    The idea is to train an auxiliary network to factorize the object related information from the 
    latent space, since we are separately feeding those information as one-hot vectors. This auxiliary 
    network job is to learn to extract the HLPs from the latent space and encoder do not want this to happen.
    as a result the auxiliary network will play an adversarial game with the encoder.

    Intent: to factorize the latent space

    Validation: We can check the if the generator output is dependent on the factorized part of the latent vector.
    """
    def __init__(self, latent_size=100, num_objects=10, num_describtions=10):
        super(Detector, self).__init__(
            g0=L.Linear(latent_size , 1000, initialW=Normal(0.02)),
            g1=L.Linear(1000, num_objects, initialW=Normal(0.02)),
            g2=L.Linear(1000, num_describtions, initialW=Normal(0.02)),
        )

    def __call__(self, z, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            h0 = self.g0(z)
            return self.g1(h0), self.g2(h0)




