import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
from chainer.initializers import Normal

class Vgg_pretrained(chainer.Chain):
    def __init__(self, latent_size=16):
        super(Vgg_pretrained, self).__init__(
            vgg = L.VGG16Layers(),
            fc6 = L.Linear(512 * 7 * 7, 4096, initialW=Normal(0.02)),
            fc7 = L.Linear(4096, 4096, initialW=Normal(0.02)),
            mean=L.Linear(4096, latent_size, initialW=Normal(0.02)),
            var=L.Linear(4096, latent_size, initialW=Normal(0.02)),
            fc7_att = L.Linear(4096, 4096, initialW=Normal(0.02)),
            mean_att = L.Linear(4096, latent_size, initialW=Normal(0.02)),
            var_att=L.Linear(4096, latent_size, initialW=Normal(0.02)),
        )
        self.vgg.disable_update() 

    def __call__(self, x, att=False, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            conv5_3 = self.vgg(x, layers=['pool5'])['pool5']
            h0 = F.leaky_relu(self.fc6(conv5_3))
            h0 = F.dropout(h0, ratio=0.5)
            if not att:
                h1 = F.leaky_relu(self.fc7(h0))
                h1 = F.dropout(h1, ratio=0.5)
                mean = self.mean(h1)
                var = F.tanh(self.var(h1))
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

class Attention(chainer.Chain):
    def __init__(self, density=1, size=28, latent_size=16, channel=1):
        assert (size % 4 == 0)
        initial_size = size / 4
        super(Attention, self).__init__(
            dc1=L.Convolution2D(channel, int(8 * density), 4, stride=2, pad=1, initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(8 * density), int(16 * density), 4, stride=2, pad=1, initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(16 * density)),
            mean=L.Linear(initial_size * initial_size * int(16 * density), latent_size, initialW=Normal(0.02)),
            var=L.Linear(initial_size * initial_size *  int(16 * density), latent_size, initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.get_array_module(x.data)
            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            mean = self.mean(h2)
            var = F.tanh(self.var(h2))
            rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            z = mean + F.exp(var) * Variable(rand)
            # z  = mean + F.exp(var) * Variable(rand, volatile=not train)
            return z, mean, var

class Encoder_text_tower(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3, hidden_dim=100, num_objects=10, num_describtions=10):
        assert (size % 16 == 0)
        second_size = size / 4
        initial_size = size / 16
        super(Encoder_text_tower, self).__init__(
            dc1=L.Convolution2D(channel, int(16 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(32 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc1_=L.Convolution2D(int(16 * density), int(16 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            dc2_=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(32 * density)),
            norm2_=L.BatchNormalization(int(32 * density)),
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
            h3_p = F.leaky_relu(self.norm3_p(self.dc3_p(h3)))
            h3_ = h3_ + h3_p
            h4 = F.leaky_relu(self.norm4(self.dc4(h3_)))
            mean = self.mean(h4)
            var = F.tanh(self.var(h4))
            rand = xp.random.normal(0, 1, var.data.shape).astype(np.float32)
            z = mean + F.exp(var) * Variable(rand)
            return z, mean, var

class Encoder_text(chainer.Chain):
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

class Encoder(chainer.Chain):
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
            # z  = mean + F.exp(var) * Variable(rand, volatile=not train)
            return z, mean, var

class Generator(chainer.Chain):
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

class Generator_text_old(chainer.Chain):
    def __init__(self, density=1, size=64, latent_size=100, channel=3, num_objects=10, num_describtions=10):
        filter_size = 4
        # intermediate_size = 256
        assert (size % 16 == 0)
        initial_size = size / 16
        super(Generator_text_old, self).__init__(
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

class Generator_text(chainer.Chain):
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
            g2_=L.Deconvolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                 initialW=Normal(0.02)),
            norm2_=L.BatchNormalization(int(64 * density)),
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
            h2_ = h2_ + h2_p
            h2_ = F.concat((h2_, h0), axis=1)
            h3 = F.relu(self.norm3(self.g3(h2_)))
            h3_ = F.relu(self.norm3_(self.g3_(h3)))
            h3_p = F.relu(self.norm3_p(self.g3_p(h3_)))
            h3_ = h3_ + h3_p
            h4 = F.relu(self.norm4(self.g4(h3_)))
            h4_ = F.relu(self.norm4_(self.g4_(h4)))
            return F.tanh(self.g5(h4_))

class Detector(chainer.Chain):
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

# class Generator_text(chainer.Chain):
#     def __init__(self, density=1, size=64, latent_size=100, channel=3, num_objects=10, num_describtions=10, num_stages=10):
#         filter_size = 4
#         # intermediate_size = 256
#         assert (size % 16 == 0)
#         initial_size = size / 16
#         super(Generator_text, self).__init__(
#             g0=L.Linear(num_objects + num_describtions, initial_size * initial_size * int(128 * density), initialW=Normal(0.02)),
#             g00=L.Linear(num_stages, initial_size * initial_size * int(128 * density), initialW=Normal(0.02)),
#             g1=L.Linear(latent_size, initial_size * initial_size * int(128 * density),
#                         initialW=Normal(0.02)),
#             norm1=L.BatchNormalization(initial_size * initial_size * int(128 * density)),
#             g2=L.Deconvolution2D(int(128 * density), int(64 * density), filter_size, stride=2, pad=1,
#                                  initialW=Normal(0.02)),
#             g21=L.Deconvolution2D(int(128 * density), int(64 * density), filter_size, stride=2, pad=1,
#                                  initialW=Normal(0.02)),
#             g22=L.Deconvolution2D(int(128 * density), int(64 * density), filter_size, stride=2, pad=1,
#                                  initialW=Normal(0.02)),
#             norm2=L.BatchNormalization(int(64 * density)),
#             norm21=L.BatchNormalization(int(64 * density)),
#             norm22=L.BatchNormalization(int(64 * density)),
#             g3=L.Deconvolution2D(int(64 * density), int(32 * density), filter_size, stride=2, pad=1,
#                                  initialW=Normal(0.02)),
#             norm3=L.BatchNormalization(int(32 * density)),
#             g4=L.Deconvolution2D(int(32 * density), int(16 * density), filter_size, stride=2, pad=1,
#                                  initialW=Normal(0.02)),
#             norm4=L.BatchNormalization(int(16 * density)),
#             g5=L.Deconvolution2D(int(16 * density), channel, filter_size, stride=2, pad=1,
#                                  initialW=Normal(0.02)),
#         )
#         self.density = density
#         self.latent_size = latent_size
#         self.initial_size = initial_size

#     def __call__(self, z, objects, descs, stage, train=True):
#         with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
#             # h0 = self.g0(F.concat((z, objects, descs), axis=-1))
#             obj_feature = self.g0(F.concat((objects, descs), axis=-1))
#             obj_feature = F.reshape(obj_feature, (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
#             stage = self.g00(stage)
#             stage_feature = F.reshape(stage, (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
#             h1 = F.reshape(F.relu(self.norm1(self.g1(z))), (z.data.shape[0], int(128 * self.density), self.initial_size, self.initial_size))
#             h2 = F.relu(self.norm2(self.g2(h1)))
#             h21 = F.relu(self.norm21(self.g21(obj_feature)))
#             h22 = F.relu(self.norm22(self.g22(stage_feature)))
#             h2 = h22 + h21 + h2
#             h3 = F.relu(self.norm3(self.g3(h2)))
#             h4 = F.relu(self.norm4(self.g4(h3)))
#             return F.tanh(self.g5(h4))


class seq_Discriminator(chainer.Chain):
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

class Discriminator_cond(chainer.Chain):
    def __init__(self, density=1, size=64, channel=3, num_objects=10, num_describtions=10):
        assert (size % 16 == 0)
        second_size = size / 4
        initial_size = size / 16
        super(Discriminator_cond, self).__init__(
            dc1=L.Convolution2D(channel, int(16 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(32 * density), 3, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc1_=L.Convolution2D(int(16 * density), int(16 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            dc2_=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(32 * density)),
            norm2_=L.BatchNormalization(int(32 * density)),
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

class Discriminator_texual_edges(chainer.Chain):
    def __init__(self, density=1, size=64, channel=1, num_objects=10):
        assert (size % 16 == 0)
        self.num_objects = num_objects
        initial_size = size / 16
        super(Discriminator_texual_edges, self).__init__(
            dc1=L.Convolution2D(channel, int(16 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            dc2=L.Convolution2D(int(16 * density), int(32 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm2=L.BatchNormalization(int(32 * density)),
            dc2_p=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2_p=L.BatchNormalization(int(32 * density)),
            dc3=L.Convolution2D(int(32 * density), int(64 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
            dc3_p=L.Convolution2D(int(64 * density), int(64 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm3_p=L.BatchNormalization(int(64 * density)),
            dc4=L.Convolution2D(int(64 * density), int(128 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm4=L.BatchNormalization(int(128 * density)),
            dc5=L.Linear(initial_size * initial_size * int(128 * density), num_objects, initialW=Normal(0.02)),
            dc8=L.Linear(initial_size * initial_size * int(128 * density), num_objects, initialW=Normal(0.02)),
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
                return self.dc5(h4), h3
            else:
                return self.dc8(h4), h3

class Discriminator_texual(chainer.Chain):
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
            dc2_p=L.Convolution2D(int(32 * density), int(32 * density), 3, stride=1, pad=1,
                                initialW=Normal(0.02)),
            norm2_p=L.BatchNormalization(int(32 * density)),
            dc3=L.Convolution2D(int(32 * density), int(64 * density), 4, stride=2, pad=1,
                                initialW=Normal(0.02)),
            norm3=L.BatchNormalization(int(64 * density)),
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

class DiscriminatorLatent(chainer.Chain):
    def __init__(self, density=1, size=64, channel=3, latent_size=128):
        assert (size % 16 == 0)
        initial_size = size / 16
        super(DiscriminatorLatent, self).__init__(
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
            dc6=L.Linear(initial_size * initial_size * int(128 * density), latent_size,
                         initialW=Normal(0.02)),
        )

    def __call__(self, x, train=True):
        with chainer.using_config('train', train):
            h1 = F.leaky_relu(self.dc1(x))
            h2 = F.leaky_relu(self.norm2(self.dc2(h1)))
            h3 = F.leaky_relu(self.norm3(self.dc3(h2)))
            h4 = F.leaky_relu(self.norm4(self.dc4(h3)))
            return self.dc5(h4), self.dc6(h4)


