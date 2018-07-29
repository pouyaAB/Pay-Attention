import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.initializers import Normal

class seq_Discriminator(chainer.Chain):
    """
    A discriminator for classifiying joint sequences to real/fake.

    It contains an LSTM layer which receives the temporal information in order and a Fully-connected layer
    which receives the hidden state from the lstm and output a 2 dimensional vector.
    """
    def __init__(self, in_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        # assert (size % 16 == 0)
        super(seq_Discriminator, self).__init__(
            ln1_=L.LayerNormalization(),
            l1_=L.LSTM(in_dim, hidden_dim),
            dc5=L.Linear(hidden_dim, 2, initialW=Normal(0.02)),
        )

    def reset_state(self):
        self.l1_.reset_state()

    def __call__(self, x, one_hot, train=True):
        with chainer.using_config('train', train):
            self.reset_state()
            sequence_len = x.shape[1]
            for j in range(sequence_len):
                input = F.concat((x[:, j], one_hot), axis=-1)
                input = self.ln1_(input)
                h0 = self.l1_(input)

            return self.dc5(h0), h0