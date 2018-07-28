import numpy as np
import math
import chainer
from chainer import training, optimizers, serializers, utils, datasets, iterators, report
from chainer import cuda, Variable, Link, Chain, ChainList
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

from gpu import GPU


class MDN_RNN(Chain):
    """
    To be renamed: RobotController
    TODO: separate the inputs as 
       image_encoding
       task_encoding 

    A network that takes as input an encoded task, an encoding z of the current camera image
    and generates the next robot angles. 

    The task is something like "pick the red towel". It is encoded as a set of 
    one hot vectors. In our current experiments we use: movement, main object shape and main object color.

    It implemented as three layers of LSTM with skip connections, layer normalization followed by an
    MDN at the output. It is trained using behavioral cloning loss. 

    """

    def __init__(self, in_dim, hidden_dim, out_dim, num_mixture, auto_regressive=False):
        self.sampling_bias = 1
        self.IN_DIM = in_dim
        self.HIDDEN_DIM = hidden_dim
        self.OUT_DIM = out_dim
        self.NUM_MIXTURE = num_mixture
        self.AUTO_REGRESSIVE = auto_regressive
        if auto_regressive:
            self.future_out_dim = 1
        else:
            self.future_out_dim = out_dim
        super(MDN_RNN, self).__init__(
            ln1_=L.LayerNormalization(),
            ln2_=L.LayerNormalization(),
            ln3_=L.LayerNormalization(),
            l1_=L.LSTM(in_dim + 4, hidden_dim),
            l2_=L.LSTM(hidden_dim + in_dim + 4, hidden_dim),
            l3_=L.LSTM(hidden_dim + in_dim + 4, hidden_dim),
            # FC1_=L.Linear(hidden_dim, self.future_out_dim)

            mixing_=L.Linear(3 * hidden_dim, num_mixture),
            mu_=L.Linear(3 * hidden_dim, num_mixture * self.future_out_dim),
            sigma_=L.Linear(3 * hidden_dim, num_mixture)
        )

    def reset_state(self):
        self.l1_.reset_state()
        self.l2_.reset_state()
        self.l3_.reset_state()

    def __call__(self, data_in=None, z=None, data_out=None, return_sample=False, train=True):

        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.cupy
            sequence_len = len(z)
            # batch_h = Variable(xp.empty((0, np.shape(data_in)[1], self.HIDDEN_DIM), dtype=np.float32))

            # data_in = F.expand_dims(data_in, axis=0)
            # data_in = F.tile(data_in, (sequence_len, 1, 1))
            # x = F.concat((data_in, z), axis=2)
            x = z
            y = data_out
            
            if self.AUTO_REGRESSIVE:
                y = F.swapaxes(y, 0, 1)
                y = F.reshape(y, (y.shape[0], y.shape[1] * y.shape[2], 1))
                y = F.swapaxes(y, 0, 1)

                x = F.swapaxes(x, 0, 1)
                x = F.tile(x, (1, self.OUT_DIM, 1))
                x = F.swapaxes(x, 0, 1)
                sequence_len = x.shape[0]

            cost = 0
            last_joint = Variable(xp.zeros(y.shape[1:], dtype=np.float32))
            sample_toRet = np.zeros(self.OUT_DIM, dtype=np.float32)
            for j in range(sequence_len):
                batch_cost, sample = self.process_batch(x[j], y[j], data_in, last_joint, return_sample=True)
                last_joint = sample
                if self.AUTO_REGRESSIVE:
                    sample_toRet[j % self.OUT_DIM] = cuda.to_cpu(sample.data)[0, 0]
                cost += batch_cost

            cost /= x.shape[0]
            if self.AUTO_REGRESSIVE:
                sample = sample_toRet
            return cost, sample

        # F.get_item(batch_h, range(1, batch_h.shape[0])), 

    def process_batch(self, x, y, one_hot, last_joint, return_sample=False, return_mean=True):
        xp = cuda.cupy
        x = F.dropout(x, ratio=0.5)
        x = F.concat((one_hot, x), axis=-1)
        x = self.ln1_(x)
        h0 = self.l1_(x)
        h = F.dropout(h0, ratio=0.5)
        h1 = self.ln2_(F.concat((x, h), axis=1))
        h1 = self.l2_(h1)
        h = F.dropout(h1, ratio=0.5)
        h2 = self.ln3_(F.concat((x, h), axis=1))
        h2 = self.l3_(h2)

        final_h = F.concat((h0, h1, h2), axis=-1)
        mu = self.mu_(final_h)
        mu = F.reshape(mu, (-1, self.future_out_dim, self.NUM_MIXTURE))

        sigma_orig = self.sigma_(final_h)
        mixing_orig = self.mixing_(final_h)

        sigma = F.softplus(sigma_orig)
        mixing = F.softmax(mixing_orig)
        
        y = F.expand_dims(y, axis=2)
        y_broad = F.broadcast_to(y, mu.shape)
        normalizer = 2 * np.pi * sigma
        exponent = -0.5 * (1. / F.square(sigma)) * F.sum((y_broad - mu) ** 2, axis=1) + F.log(mixing) - (self.future_out_dim * 0.5) * F.log(normalizer)
        # if not flag:
        cost = -F.logsumexp(exponent, axis=1)
        cost = F.mean(cost)
        # else:
        #     max_exponent = F.max(exponent, axis=1, keepdims=True)
        #     mod_exponent = exponent - F.broadcast_to(max_exponent, exponent.shape)
        #     gauss_mix = F.sum(F.exp(mod_exponent), axis=1, keepdims=True)
        #     log_gauss = F.log(gauss_mix) + max_exponent
            
        #     cost = -F.mean(log_gauss * y[:, -1])
        # cost = F.mean_squared_error(y, mu)
        #sampling
        if return_sample:
            mixing = mixing_orig * (1 + self.sampling_bias)
            sigma = F.softplus(sigma_orig - self.sampling_bias)
            mixing = F.softmax(mixing)
            # max_mixing = F.broadcast_to(F.max(mixing, axis=2, keepdims=True), mixing.shape)
            # e_x = F.exp(mixing - max_mixing)
            # e_x_sum = F.broadcast_to(F.sum(e_x, axis=2, keepdims=True), e_x.shape)
            # mixing = e_x / e_x_sum
            # component = xp.zeros(mixing.shape, dtype=xp.float32)
            # for i in range(mixing.shape[0]):
            #     component[i] = xp.random.multinomial(1, pvals=mixing[i].data)
            # component_expanded = F.broadcast_to(F.expand_dims(component, axis=1), mu.shape)
            argmax_mixing = F.argmax(mixing, axis=1)
            mixing_one_hot = xp.zeros(mixing.shape, dtype=xp.float32)
            mixing_one_hot[xp.arange(mixing.shape[0]), argmax_mixing.data] = 1
            component_expanded = F.broadcast_to(F.expand_dims(mixing_one_hot, axis=1), mu.shape)
            component_mean = F.sum(mu * component_expanded, axis=2)
            if return_mean:
                return cost, component_mean
            component_std = F.sum(sigma * component, axis=2, keepdims=True)
            component_std = F.broadcast_to(component_std, component_mean.shape)

            sample = xp.random.normal(component_mean.data, component_std.data)
            return cost, sample
        
        return cost, None

