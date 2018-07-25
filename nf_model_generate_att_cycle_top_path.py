import numpy as np
import time
import sys
import os
import copy
import chainer.functions as F
from PIL import Image
import threading
import signal
import copy

from matplotlib.pyplot import margins

from gpu import GPU
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
import autoencoder
from nf_mdn_rnn import MDN_RNN
from Utilities import Utils
# from nf_roboinstruct_dataset_controller import DatasetController
# from nf_movingMNIST_dataset_controller import DatasetController
from DatasetController_morph import DatasetController
from DatasetController_multi_object_morph import DatasetController_multi_object

from local_config import config

locals().update(config)


def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

class ModelController:
    """
        Handles the model interactions
    """

    def __init__(self, image_size, latent_size, batch_size, sequence_size,
                 num_channels=3, save_dir="model/", epoch_num=320,
                 sample_image_cols=8, sample_image_rows=5, load_models=True):
        # self.dataset_ctrl = DatasetController(batch_size=batch_size, sequence_size=sequence_size,
        #                                       image_size=(image_size, image_size), dataset_path=dataset_path)

        self.dataset_ctrl = DatasetController(batch_size=int(batch_size/2), sequence_input=sequence_size, sequence_output=0, read_jpgs=True)
        self.dataset_multi_ctrl = DatasetController_multi_object(batch_size=int(batch_size/2), sequence_size=1)
        self.num_tasks = len(config['tasks'])
        self.ae_mode = 'VAE'  # AAE, VAE
        self.image_size = image_size
        self.normer = image_size * image_size * 3 * 60
        self.vocabularySize = 34
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.num_channels = num_channels
        self.latent_size = latent_size
        self.hidden_dimension = hidden_dimension
        self.num_mixture = num_mixture
        self.output_size = output_size
        self.save_dir = save_dir
        self.epoch_num = epoch_num
        self.train_lstm_alone = False
        self.train_lstm = False or self.train_lstm_alone
        self.train_autoencoder = True and not self.train_lstm_alone
        self.train_dis = False and not self.train_lstm_alone
        self.load_models = load_models
        self.last_best_result = 100
        self.save_model_period = 501
        self.save_sample_image_interval = 501

        self.sample_image_cols = sample_image_cols
        self.sample_image_rows = sample_image_rows
        self.generator_test = self.dataset_ctrl.get_next_batch(task=['5002', '5001'], from_start_prob=0, human=False, train=False, sth_sth=False, camera='camera-1')
        self.generator_test_multi = self.dataset_multi_ctrl.get_next_batch(train=False, camera='camera-1')

        self.num_descs = self.dataset_ctrl.num_all_objects_describtors + 1
        self.num_objects = self.dataset_ctrl.num_all_objects + 1

        images_att, images, cropped_images, _, _, _, _, _, _, _, _ , _, _, _, objects, objs_one_hot, describtions, descs_one_hot, _, _ = next(self.generator_test)
        images_att_multi, images_multi, cropped_images_multi, _, objs_multi, objs_one_hot_multi, descs_multi, descs_one_hot_multi, _, _ = next(self.generator_test_multi)

        images = np.reshape(images, (-1, self.num_channels, self.image_size, self.image_size))
        images_att = np.reshape(images_att, (-1, self.num_channels, self.image_size, self.image_size))
        images_multi = np.reshape(images_multi, (-1, self.num_channels, self.image_size, self.image_size))
        images_att_multi = np.reshape(images_att_multi, (-1, self.num_channels, self.image_size, self.image_size))
        cropped_images = np.reshape(cropped_images, (-1, self.num_channels, self.image_size, self.image_size))
        cropped_images_multi = np.reshape(cropped_images_multi, (-1, self.num_channels, self.image_size, self.image_size))

        objs_one_hot_multi = np.squeeze(objs_one_hot_multi)
        descs_one_hot_multi = np.squeeze(descs_one_hot_multi)
        objs_multi = np.squeeze(objs_multi)
        descs_multi = np.squeeze(descs_multi)

        # np.random.shuffle(images)
        sample_size = self.sample_image_cols * self.sample_image_rows / 2
        temp = np.asarray(images[:sample_size], dtype=np.float32)
        self.sample_images = np.concatenate((temp, np.asarray(images_multi[:sample_size], dtype=np.float32)), axis=0)
        temp = np.asarray(images_att[:sample_size], dtype=np.float32)
        self.sample_images_att = np.concatenate((temp, np.asarray(images_att_multi[:sample_size], dtype=np.float32)), axis=0)
        temp = np.asarray(cropped_images[:sample_size], dtype=np.float32)
        self.sample_images_crop = np.concatenate((temp, np.asarray(cropped_images_multi[:sample_size], dtype=np.float32)), axis=0)
        temp = np.asarray(objs_one_hot[:sample_size], dtype=np.float32)
        self.sample_objs_one_hot = np.concatenate((temp, np.asarray(objs_one_hot_multi[:sample_size], dtype=np.float32)), axis=0)
        temp = np.asarray(descs_one_hot[:sample_size], dtype=np.float32)
        self.sample_descs_one_hot = np.concatenate((temp, np.asarray(descs_one_hot_multi[:sample_size], dtype=np.float32)), axis=0)

        temp = np.asarray(objects[:sample_size], dtype=np.float32)
        objects = np.concatenate((temp, np.asarray(objs_multi[:sample_size], dtype=np.float32)), axis=0)
        temp = np.asarray(describtions[:sample_size], dtype=np.float32)
        describtions = np.concatenate((temp, np.asarray(descs_multi[:sample_size], dtype=np.float32)), axis=0)

        # sample_size = self.sample_image_cols * self.sample_image_rows
        # self.sample_images = np.asarray(images[:sample_size], dtype=np.float32)
        # self.sample_images_att = np.asarray(images_att[:sample_size], dtype=np.float32)
        # self.sample_objs_one_hot = np.asarray(objs_one_hot[:sample_size], dtype=np.float32)
        # self.sample_descs_one_hot = np.asarray(descs_one_hot[:sample_size], dtype=np.float32)
        # objects = np.asarray(objects[:sample_size], dtype=np.float32)
        # describtions = np.asarray(describtions[:sample_size], dtype=np.float32)

        objects = (objects / self.num_objects) - 1
        objects = objects[:, np.newaxis, np.newaxis, np.newaxis]
        self.sample_objects_norm = np.tile(objects, (1, 1, self.image_size, self.image_size))
        describtions = (describtions / self.num_descs) - 1
        describtions = describtions[:, np.newaxis, np.newaxis, np.newaxis]
        self.sample_describtions_norm = np.tile(describtions, (1, 1, self.image_size, self.image_size))

        self.generator = self.dataset_ctrl.get_next_batch(task=['5002', '5001'], from_start_prob=0, human=False, sth_sth=False, camera='camera-1')
        self.generator_multi = self.dataset_multi_ctrl.get_next_batch(train=True, camera='camera-1')
        print objs_multi[:sample_size]
        print descs_multi[:sample_size]

        self.enc_model = autoencoder.Encoder_text_tower(density=8, size=image_size, latent_size=latent_size, channel=self.num_channels, num_objects=self.num_objects - 1, num_describtions=self.num_descs - 1)
        self.gen_model = autoencoder.Generator_text(density=8, size=image_size, latent_size=latent_size, channel=self.num_channels * 2, num_objects=self.num_objects - 1, num_describtions=self.num_descs - 1)
        # self.enc_model_att = autoencoder.Encoder_text_tower(density=8, size=image_size, latent_size=latent_size, channel=self.num_channels, num_objects=self.num_objects - 1, num_describtions=self.num_descs - 1)
        # self.gen_model_att = autoencoder.Generator_text(density=8, size=image_size, latent_size=latent_size, channel=self.num_channels, num_objects=self.num_objects - 1, num_describtions=self.num_descs - 1)
        self.dis_model = autoencoder.Discriminator_texual(density=8, size=image_size, channel=self.num_channels,
                                                         num_words=self.vocabularySize, num_objects=self.num_objects, num_describtions=self.num_descs)
        # self.dis_model_edges = autoencoder.Discriminator_texual_edges(density=8, size=image_size, channel=1, num_objects=self.num_objects)
        self.det_model = autoencoder.Detector(latent_size=latent_size, num_objects=self.num_objects, num_describtions=self.num_descs)
        # self.det_model_att = autoencoder.Detector(latent_size=latent_size, num_objects=self.num_objects, num_describtions=self.num_descs)

        self.enc_models = [self.enc_model]
        self.gen_models = [self.gen_model]
        # self.enc_models_att = [self.enc_model_att]
        # self.gen_models_att = [self.gen_model_att]
        self.dis_models = [self.dis_model]
        # self.dis_models_edges = [self.dis_model_edges]
        self.det_models = [self.det_model]
        # self.det_models_att = [self.det_model_att]

        self.learning_rate = 0.0001
        self.WeightDecay = 0.00001

        self.optimizer_enc = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        self.optimizer_enc.setup(self.enc_models[0])
        self.optimizer_enc.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        self.optimizer_gen = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        self.optimizer_gen.setup(self.gen_models[0])
        self.optimizer_gen.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        # self.optimizer_enc_att = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        # self.optimizer_enc_att.setup(self.enc_models_att[0])
        # self.optimizer_enc_att.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        # self.optimizer_gen_att = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        # self.optimizer_gen_att.setup(self.gen_models_att[0])
        # self.optimizer_gen_att.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        self.optimizer_dis = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        self.optimizer_dis.setup(self.dis_models[0])
        self.optimizer_dis.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        # self.optimizer_dis_edges = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        # self.optimizer_dis_edges.setup(self.dis_models_edges[0])
        # self.optimizer_dis_edges.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        self.optimizer_det = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        self.optimizer_det.setup(self.det_models[0])
        self.optimizer_det.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        # self.optimizer_det_att = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        # self.optimizer_det_att.setup(self.det_models_att[0])
        # self.optimizer_det_att.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        for i in range(GPU.num_gpus - 1):
            self.enc_models.append(copy.deepcopy(self.enc_model))
            self.gen_models.append(copy.deepcopy(self.gen_model))
            # self.enc_models_att.append(copy.deepcopy(self.enc_model_att))
            # self.gen_models_att.append(copy.deepcopy(self.gen_model_att))
            self.dis_models.append(copy.deepcopy(self.dis_model))
            # self.dis_models_edges.append(copy.deepcopy(self.dis_model_edges))
            self.det_models.append(copy.deepcopy(self.det_model))
            # self.det_models_att.append(copy.deepcopy(self.det_model_att))

        self.batch_gpu_threads = [None] * GPU.num_gpus

        # self.models = [self.enc_model, self.gen_model, self.gen_mask_model, self.dis_model, self.aae_dis_model]
        # self.models = [self.enc_model, self.gen_model, self.dis_model]
        if self.load_models:
            self.load_model()
        self.to_gpu()
        self.create_sample_folders()

    def create_sample_folders(self):
        if not os.path.exists('sample/att'):
            os.makedirs('sample/att')
        if not os.path.exists('sample/whole'):
            os.makedirs('sample/whole')
        if not os.path.exists('sample/e_whole'):
            os.makedirs('sample/e_whole')
        if not os.path.exists('sample/e_att'):
            os.makedirs('sample/e_att')

    def reset_all(self, models):
        for model in models:
            model.reset_state()

    def clear_grads(self, models):
        for model in models:
            model.cleargrads()

    def show_image(self, images):
        # for i in range(images.shape[0]):
            # moving axis to use plt: i.e [4,100,100] to [100,100,4]
        img = images
        img = img.transpose(1, 2, 0)
        img = (img + 1) * 127.5
        img = img.astype(np.uint8)
        print img.dtype, np.max(img), np.min(img), np.shape(img)
        img = Image.fromarray(img, "RGB")
        img.show()

    def mix(self, a, b):
        sh = a.shape
        sh = (sh[0] * 2,) + a.shape[1:]
        c = np.empty(sh, dtype=a.dtype)
        c[0::2] = a
        c[1::2] = b
        return c

    def conv_2d_kernel(self, image_array_2d, g):
        k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        k = k[np.newaxis, np.newaxis]
        k = np.repeat(k, 3, axis=1)
        kernel = Variable(cuda.to_gpu(k, g))
        transformed_array = F.convolution_2d(image_array_2d, kernel, pad=1)
        transformed_array = F.clip(transformed_array, -1.0, 1.0)
        return transformed_array

    def train(self):
        xp = cuda.cupy
        cuda.get_device(GPU.main_gpu).use()

        self.save_sample_images(epoch=0, batch=0)
        for epoch in xrange(1, self.epoch_num + 1):
            print '\n ------------- epoch {0} started ------------'.format(epoch)
            batches_passed = 0
            while batches_passed < 1000:
                batches_passed += 1
                batch_start_time = time.time()
                # data_in, data_out, images = self.dataset_ctrl.get_next_batch(no_seq=not self.train_lstm)
                att_images, images, _, _, _, _, _, _, _, _, _, _, _, _, objs, objs_one_hot, descs, descs_one_hot, _, _ = next(self.generator)
                att_images_multi, images_multi, _, _, objs_multi, objs_one_hot_multi, descs_multi, descs_one_hot_multi, _, _ = next(self.generator_multi)

                objs_one_hot_multi = np.squeeze(objs_one_hot_multi)
                descs_one_hot_multi = np.squeeze(descs_one_hot_multi)
                objs_multi = np.squeeze(objs_multi)
                descs_multi = np.squeeze(descs_multi)
                # cropped_att_images = self.mix(cropped_images, cropped_images_multi)
                att_images = self.mix(att_images, att_images_multi)
                images = self.mix(images, images_multi)
                # sentence = self.mix(sentence, sentence_multi)
                objs = self.mix(objs, objs_multi)
                objs_one_hot = self.mix(objs_one_hot, objs_one_hot_multi)
                descs = self.mix(descs, descs_multi)
                descs_one_hot = self.mix(descs_one_hot, descs_one_hot_multi)

                att_images = np.concatenate((att_images, np.copy(images)), axis=2)
                # images = np.concatenate((images, cropped_att_images), axis=2)

                # images_all_cameras, _, _, _, _, _, _, _, _, _, _, _, objs_all_cameras, _, descs_all_cameras, _ = next(self.generator_disc)
                # self.show_image(images[0,0])
                # gt_vocab = np.zeros((self.batch_size, self.vocabularySize))
                # for b in range(self.batch_size):
                #     for w in sentence[b,:]:
                #         gt_vocab[b,int(w)] = 1
                # images = np.concatenate((images, human_igmages, sth_sth_images), axis=0)
                # np.random.shuffle(images)
                # image = ((cuda.to_cpu(images[:, 5]) + 1) * 128).clip(0, 255).astype(np.uint8)
                # # image = image[:self.sample_image_rows * self.sample_image_cols]
                # image = image.reshape(
                #     (1, self.sequence_size, self.num_channels, self.image_size, self.image_size)).transpose(
                #     (0, 3, 1, 4, 2)).reshape(
                #     (1 * self.image_size, self.sequence_size * self.image_size))
                # # print np.max(image), np.min(image), type(image), np.shape(image), image
                # Image.fromarray(image, mode='L').save('batch.png')

                for k, g in enumerate(GPU.gpus_to_use):
                    # self.handle_gpu_batch(batch_start_time, k, g, images, data_in, data_out)
                    self.batch_gpu_threads[k] = threading.Thread(target=self.handle_gpu_batch, args=(
                        batches_passed, batch_start_time, k, g, images, att_images, None, None, None, None, objs, objs_one_hot, descs, descs_one_hot))
                    self.batch_gpu_threads[k].start()

                for i in range(GPU.num_gpus):
                    self.batch_gpu_threads[i].join()
                self.add_grads()

                if self.train_autoencoder:
                    self.optimizer_enc.update()
                    self.optimizer_gen.update()
                    self.optimizer_det.update()
                    # self.optimizer_enc_att.update()
                    # self.optimizer_gen_att.update()

                if self.train_dis:  # float(loss_dis.data) > 0.0001:
                    self.optimizer_dis.update()
                    # self.optimizer_dis_edges.update()

                # self.optimizer_det_att.update()

                self.copy_params()

                current_batch = batches_passed
                if current_batch % self.save_sample_image_interval == 0:
                    self.save_sample_images(epoch=epoch, batch=current_batch)

                if current_batch % self.save_model_period == self.save_model_period - 1:
                    self.save_models()

            self.save_models()
            self.save_sample_images(epoch=epoch, batch=batches_passed)

    def rnd_categorical(self, n, n_categorical):
        indices = np.random.randint(n_categorical, size=n)
        one_hot = np.zeros((n, n_categorical))
        one_hot[np.arange(n), indices] = 1
        return np.asarray(one_hot, dtype=np.float32), indices
    

    def handle_gpu_batch(self, batches_passed, batch_start_time, k, g, images, att_images, secondary_images, joints, batch_one_hot, gt_vocab, objects, objs_one_hot, describtions, describtions_one_hot):
        xp = cuda.cupy
        cuda.get_device(g).use()
        self.dis_models[k].cleargrads()
        # self.dis_models_edges[k].cleargrads()
        self.gen_models[k].cleargrads()
        self.enc_models[k].cleargrads()
        # self.gen_models_att[k].cleargrads()
        # self.enc_models_att[k].cleargrads()
        self.det_models[k].cleargrads()
        # self.det_models_att[k].cleargrads()
        # self.reset_all([self.mdn_models[k]])
        gpu_batch_size = self.batch_size // GPU.num_gpus
        images = images.transpose(1, 0, 2, 3, 4)
        images = images[:, k * gpu_batch_size:(k + 1) * gpu_batch_size]
        att_images = att_images.transpose(1, 0, 2, 3, 4)
        att_images = att_images[:, k * gpu_batch_size:(k + 1) * gpu_batch_size]
        #secondary_images = secondary_images.transpose(1, 0, 2, 3, 4)
        #secondary_images = secondary_images[:, k * gpu_batch_size:(k + 1) * gpu_batch_size]
        objects_norm = np.asarray(objects[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        describtions_norm = np.asarray(describtions[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        objects = np.asarray(objects[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.int32)
        describtions = np.asarray(describtions[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.int32)
        real_objects = objects - 1
        real_describtions = describtions - 1
        objects_norm = (objects_norm / self.num_objects) - 1
        objects_norm = objects_norm[:, np.newaxis, np.newaxis, np.newaxis]
        objects_norm = np.tile(objects_norm, (1, 1, self.image_size, self.image_size))
        describtions_norm = (describtions_norm / self.num_descs) - 1
        describtions_norm = describtions_norm[:, np.newaxis, np.newaxis, np.newaxis]
        describtions_norm = np.tile(describtions_norm, (1, 1, self.image_size, self.image_size))
        # completeness = np.asarray(completeness[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.int32)
        objects_one_hot = np.asarray(objs_one_hot[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        descs_one_hot = np.asarray(describtions_one_hot[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        # completeness_one_hot = np.asarray(completeness_one_hot[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        # gt_vocab_batch = np.asarray(gt_vocab[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.int32)
        images = np.reshape(images, (-1, self.num_channels, self.image_size, self.image_size))
        x_in = cuda.to_gpu(np.asarray(images, dtype=np.float32), g)
        att_images = np.reshape(att_images, (-1, self.num_channels * 2, self.image_size, self.image_size))
        x_in_att = cuda.to_gpu(np.asarray(att_images, dtype=np.float32), g)
        #secondary_images = np.reshape(secondary_images, (-1, self.num_channels, self.image_size, self.image_size))
        #x_in_secondary = cuda.to_gpu(np.asarray(secondary_images, dtype=np.float32), g)

        # joints = joints.transpose(1, 0, 2)
        # joints = Variable(cuda.to_gpu(joints[:, k * gpu_batch_size:(k + 1) * gpu_batch_size], g))

        # batch_one_hot = Variable(cuda.to_gpu(batch_one_hot[k * gpu_batch_size:(k + 1) * gpu_batch_size], g))
        objects_var = Variable(cuda.to_gpu(objects, g))
        desc_var = Variable(cuda.to_gpu(describtions, g))

        real_objects_var = Variable(cuda.to_gpu(real_objects, g))
        real_desc_var = Variable(cuda.to_gpu(real_describtions, g))

        gpu_images_size = len(images)

        z0, mean, var = self.enc_models[k](Variable(x_in), Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)) , train=self.train_autoencoder)    
        l_prior = F.gaussian_kl_divergence(mean, var) / (2 * self.normer)
    
        detected_one_hot_obj, detected_one_hot_desc = self.det_models[k](z0, train=self.train_autoencoder)
        l_det = F.softmax_cross_entropy(detected_one_hot_obj, real_objects_var) + F.softmax_cross_entropy(detected_one_hot_desc, real_desc_var)

        x0_att = self.gen_models[k](z0, Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)), train=self.train_autoencoder)
        reconstruction_loss_att = F.mean_squared_error(x0_att[0::2, 3:], x_in_att[0::2, 3:])

        z00, mean0, var0 = self.enc_models[k](x0_att[:, 3:], Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)) , train=self.train_autoencoder)    
        l_prior += F.gaussian_kl_divergence(mean0, var0) / (2 * self.normer)

        detected_one_hot_obj0, detected_one_hot_desc0 = self.det_models[k](z00, train=self.train_autoencoder)
        l_det += F.softmax_cross_entropy(detected_one_hot_obj0, real_objects_var) + F.softmax_cross_entropy(detected_one_hot_desc0, real_desc_var)

        x0_att0 = self.gen_models[k](z00, Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)), train=self.train_autoencoder)
        reconstruction_loss_att += F.mean_squared_error(x0_att0, x0_att)
        # reconstruction_loss_whole = F.mean_squared_error(x0_whole_att[0::2], x_in[0::2])
        # reconstruction_loss_whole = F.mean_squared_error(x0_whole_att, x_in)

        # z00, mean00, var00 = self.enc_models_att[k](x0_att, Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)), train=self.train_autoencoder)
        # z00_att, mean00_att, var00_att = self.enc_models[k](x0_whole_att, Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)), train=self.train_autoencoder)
        
        # l_prior += F.gaussian_kl_divergence(mean00, var00) / (2 * self.normer)
        # l_prior += F.gaussian_kl_divergence(mean00_att, var00_att) / (2 * self.normer)

        # detected_one_hot_obj_2, detected_one_hot_desc_2 = self.det_models[k](z00_att, train=self.train_autoencoder)
        # detected_one_hot_obj_att_2, detected_one_hot_desc_att_2 = self.det_models_att[k](z00, train=self.train_autoencoder)

        # l_det += F.softmax_cross_entropy(detected_one_hot_obj_2, real_objects_var) + F.softmax_cross_entropy(detected_one_hot_desc_2, real_desc_var)
        # l_det_att += F.softmax_cross_entropy(detected_one_hot_obj_att_2, real_objects_var) + F.softmax_cross_entropy(detected_one_hot_desc_att_2, real_desc_var)

        # x00_whole = self.gen_models_att[k](z00, Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)), train=self.train_autoencoder)
        # x00_att_att = self.gen_models[k](z00_att, Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)), train=self.train_autoencoder)


        # e0_att, e_l0_att = self.dis_models_edges[k](x0_att_edges, train=self.train_autoencoder)
        # e0_whole, e_l0_whole = self.dis_models_edges[k](x0_whole_edges, att=False, train=self.train_autoencoder)

        # l_dis_edges_rec_att = F.softmax_cross_entropy(e0_att, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))
        # l_dis_edges_rec_whole = F.softmax_cross_entropy(e0_whole, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))

        # l_dis_edges_0 = (l_dis_edges_rec_att + l_dis_edges_rec_whole) / (2 * gpu_images_size)


        y0_att, d0_att, l0_att = self.dis_models[k](x0_att[:, :3], train=self.train_autoencoder)
        y0_att0, d0_att0, l0_att0 = self.dis_models[k](x0_att0[:, :3], train=self.train_autoencoder)

        y0_whole, d0_whole, l0_whole = self.dis_models[k](x0_att[:, 3:], att=False, train=self.train_autoencoder)
        y0_whole0, d0_whole0, l0_whole0 = self.dis_models[k](x0_att0[:, 3:], att=False, train=self.train_autoencoder)
        
        # y00_whole, d00_whole, l00_whole = self.dis_models[k](x00_whole, att=False, train=self.train_autoencoder)
        # y0_att2, d0_att2, l0_att2 = self.dis_models[k](x0_whole_att, att=False, train=self.train_autoencoder)
        # y00_whole2, d00_whole2, l00_whole2 = self.dis_models[k](x00_att_att, train=self.train_autoencoder)

        l_dis_obj_rec_att = F.softmax_cross_entropy(y0_att, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))
        l_dis_desc_rec_att = F.softmax_cross_entropy(d0_att, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))
        l_dis_obj_rec_att0 = F.softmax_cross_entropy(y0_att0, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))
        l_dis_desc_rec_att0 = F.softmax_cross_entropy(d0_att0, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))

        l_dis_obj_rec_whole = F.softmax_cross_entropy(y0_whole, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))
        l_dis_desc_rec_whole = F.softmax_cross_entropy(d0_whole, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))
        l_dis_obj_rec_whole0 = F.softmax_cross_entropy(y0_whole0, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))
        l_dis_desc_rec_whole0 = F.softmax_cross_entropy(d0_whole0, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))

        l_dis_0 = (l_dis_obj_rec_att + l_dis_desc_rec_att + l_dis_obj_rec_att0 + l_dis_desc_rec_att0) / (4 * gpu_images_size)
        l_dis_0 += (l_dis_obj_rec_whole + l_dis_desc_rec_whole + l_dis_obj_rec_whole0 + l_dis_desc_rec_whole0) / (4 * gpu_images_size)

        # l_gen_edges_sim_att = F.softmax_cross_entropy(e0_att, objects_var)
        # l_gen_edges_sim_whole = F.softmax_cross_entropy(e0_whole, objects_var)

        # l_gen_edges_0 = (l_gen_edges_sim_att + l_gen_edges_sim_whole) / (2 * gpu_images_size)



        l_gen_obj_sim_att = F.softmax_cross_entropy(y0_att, objects_var)
        l_gen_desc_sim_att = F.softmax_cross_entropy(d0_att, desc_var)
        l_gen_obj_sim_att0 = F.softmax_cross_entropy(y0_att0, objects_var)
        l_gen_desc_sim_att0 = F.softmax_cross_entropy(d0_att0, desc_var)

        l_gen_obj_sim_whole = F.softmax_cross_entropy(y0_whole, objects_var)
        l_gen_desc_sim_whole = F.softmax_cross_entropy(d0_whole, desc_var)
        l_gen_obj_sim_whole0 = F.softmax_cross_entropy(y0_whole0, objects_var)
        l_gen_desc_sim_whole0 = F.softmax_cross_entropy(d0_whole0, desc_var)

        l_gen_0 = (l_gen_obj_sim_att + l_gen_desc_sim_att + l_gen_obj_sim_att0 + l_gen_desc_sim_att0) / (4 * gpu_images_size)
        l_gen_0 += (l_gen_obj_sim_whole + l_gen_desc_sim_whole + l_gen_obj_sim_whole0 + l_gen_desc_sim_whole0) / (4 * gpu_images_size)



        z1 = Variable(cuda.to_gpu(xp.random.normal(0, 1, (gpu_images_size, self.latent_size), dtype=np.float32), g))
        x1_att = self.gen_models[k](z1, Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)), train=self.train_autoencoder)
        y1_att, d1_att, _ = self.dis_models[k](x1_att[:, :3], train=self.train_dis)
        y1_whole, d1_whole, _ = self.dis_models[k](x1_att[:, 3:], att=False, train=self.train_dis)

        # y1_edges_att, _ = self.dis_models_edges[k](self.conv_2d_kernel(x1_att[:, :3], g), train=self.train_dis)
        # y1_edges_whole, _ = self.dis_models_edges[k](self.conv_2d_kernel(x1_att[:, 3:], g), att=False, train=self.train_dis)

        l_dis_obj_fake_att = F.softmax_cross_entropy(y1_att, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g))) 
        l_dis_desc_fake_att = F.softmax_cross_entropy(d1_att, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))
        l_dis_obj_fake_whole = F.softmax_cross_entropy(y1_whole, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g))) 
        l_dis_desc_fake_whole = F.softmax_cross_entropy(d1_whole, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))
        l_dis_random_fake = (l_dis_obj_fake_att + l_dis_desc_fake_att + l_dis_obj_fake_whole + l_dis_desc_fake_whole) / (4 * gpu_images_size)

        # l_dis_edges_fake_att = F.softmax_cross_entropy(y1_edges_att, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g))) 
        # l_dis_edges_fake_whole = F.softmax_cross_entropy(y1_edges_whole, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g))) 
        # l_dis_edges_random_fake = (l_dis_edges_fake_att + l_dis_edges_fake_whole) / (2 * gpu_images_size)

        # y2_edges, l2_edges = self.dis_models_edges[k](x_in_edge[0::2], att=False, train=self.train_dis)
        # y2_att_edges, l2_att_edges = self.dis_models_edges[k](x_in_att_edge, train=self.train_dis)

        y2_att, d2_att, l2_att = self.dis_models[k](x_in_att[:, :3], train=self.train_dis)
        y2_whole, d2_whole, l2_whole = self.dis_models[k](x_in_att[0::2, 3:], att=False, train=self.train_dis)


        # l_dis_edges_real_att = F.softmax_cross_entropy(y2_edges, objects_var[0::2]) 
        # l_dis_edges_real_whole = F.softmax_cross_entropy(y2_att_edges, objects_var) 
        # l_dis_edges_real = (l_dis_edges_real_att + l_dis_edges_real_whole) / (2 * gpu_images_size)

        l_dis_obj_real_att = F.softmax_cross_entropy(y2_att, objects_var) 
        l_dis_desc_real_att = F.softmax_cross_entropy(d2_att, desc_var)
        l_dis_obj_real_whole = F.softmax_cross_entropy(y2_whole, objects_var[0::2]) 
        l_dis_desc_real_whole = F.softmax_cross_entropy(d2_whole, desc_var[0::2])
        l_dis_real = (l_dis_obj_real_att + l_dis_desc_real_att + l_dis_obj_real_whole + l_dis_desc_real_whole) / (4 * gpu_images_size)

        # l_feature_similarity_edges = F.mean_squared_error(e_l0_att, l2_att_edges) + F.mean_squared_error(e_l0_whole[0::2], l2_edges)

        l_feature_similarity_att = F.mean_squared_error(l0_att, l2_att) + F.mean_squared_error(l0_att0, l0_att)
        l_feature_similarity_whole = F.mean_squared_error(l0_whole[0::2], l2_whole)  + F.mean_squared_error(l0_whole0[0::2], l0_whole[0::2])
        l_feature_similarity = (l_feature_similarity_att + l_feature_similarity_whole)
        # l_feature_similarity += l_feature_similarity_edges
        reconstruction_loss = reconstruction_loss_att
        # l_feature_similarity = F.mean_squared_error(l0_att, l2_att) + F.mean_squared_error(l00_whole, l2_whole)
        # l_feature_similarity += F.mean_squared_error(l0_att2, l2_whole) + F.mean_squared_error(l00_whole2, l2_att)
        # l_dis_edges_sum = (l_dis_edges_0 + l_dis_edges_random_fake + l_dis_edges_real)
        l_dis_sum = (l_dis_random_fake + l_dis_0 + l_dis_real)
        loss_dets = l_det / gpu_images_size
        loss_enc = l_prior + l_feature_similarity
        loss_dis = l_dis_sum
        loss_gen = l_gen_0 + l_feature_similarity + reconstruction_loss - l_dis_sum

        self.train_dis = True
        self.train_autoencoder = True

        if self.train_autoencoder:
            self.enc_models[k].cleargrads()
            self.gen_models[k].cleargrads()
            self.det_models[k].cleargrads()
            # self.enc_models_att[k].cleargrads()
            # self.gen_models_att[k].cleargrads()
            loss_net = loss_enc + loss_gen
            loss_net.backward()
            # loss_enc.backward()
            # loss_gen.backward()

        x0_att.unchain()
        # x0_att_edges.unchain()
        # x0_whole_edges.unchain()
        # x00_whole.unchain()
        x1_att.unchain()
        # x12_whole.unchain()

        if self.train_dis:
            self.dis_models[k].cleargrads()
            # self.dis_models_edges[k].cleargrads()
            # loss_dises = loss_dis + l_dis_edges_sum
            loss_dises = loss_dis
            loss_dises.backward()
            # l_dis_edges_sum.backward()

        # z0.unchain()
        # z00.unchain()
        # z0_att.unchain()
        # z00_att.unchain()
        # self.det_models_att[k].cleargrads()



        sys.stdout.write('\r' + str(batches_passed) + '/' + str(1000) +
                         ' time: {0:0.2f}, enc:{1:0.4f}, gen:{2:0.4f}, dis:{3:0.4f}, gen_0:{4:0.4f}, rec:{5:0.4f}, pri:{6:0.4f}, fea:{7:0.4f} ,det_loss:{8:0.4f}'.format(
                             time.time() - batch_start_time,
                             float(loss_enc.data),
                             float(loss_gen.data),
                             float(loss_dis.data),
                             float(l_gen_0.data),
                             float(reconstruction_loss_att.data),
                             float(l_prior.data),
                             float(l_feature_similarity.data),
                             float(loss_dets.data),
                             float(0.0)
                             # float(lstm_cost.data)
                             # float(tcn_loss.data)
                         ))
        sys.stdout.flush()  # important

    def copy_params(self):
        for i in range(1, GPU.num_gpus):
            self.enc_models[i].copyparams(self.enc_models[0])
            self.gen_models[i].copyparams(self.gen_models[0])
            # self.enc_models_att[i].copyparams(self.enc_models_att[0])
            # self.gen_models_att[i].copyparams(self.gen_models_att[0])
            self.dis_models[i].copyparams(self.dis_models[0])
            # self.dis_models_edges[i].copyparams(self.dis_models_edges[0])
            self.det_models[i].copyparams(self.det_models[0])
            # self.det_models_att[i].copyparams(self.det_models_att[0])

    def add_grads(self):
        for j in range(1, GPU.num_gpus):
            self.enc_models[0].addgrads(self.enc_models[j])
            self.gen_models[0].addgrads(self.gen_models[j])
            # self.enc_models_att[0].addgrads(self.enc_models_att[j])
            # self.gen_models_att[0].addgrads(self.gen_models_att[j])
            self.dis_models[0].addgrads(self.dis_models[j])
            # self.dis_models_edges[0].addgrads(self.dis_models_edges[j])
            self.det_models[0].addgrads(self.det_models[j])
            # self.det_models_att[0].addgrads(self.det_models_att[j])

    # def sample(self, start, goal, iter_num=10):
    #     # self.load_model()
    #     xp = cuda.cupy
    #     prediction = xp.asarray(start, dtype="float32")
    #     goal = xp.asarray(goal, dtype="float32")
    #     for i in range(iter_num):
    #         print 'prediction: ', prediction
    #         start_seq = xp.concatenate((goal, prediction[:-1]))
    #         seq_in = Variable(cuda.to_gpu(xp.asarray(start_seq, dtype="float32"), GPU.main_gpu))
    #         var_predicted = self.mdn_model.predict(seq_in)
    #         prediction = var_predicted[0]

    def to_gpu(self):
        for i in range(GPU.num_gpus):
            self.enc_models[i].to_gpu(GPU.gpus_to_use[i])
            self.gen_models[i].to_gpu(GPU.gpus_to_use[i])
            # self.enc_models_att[i].to_gpu(GPU.gpus_to_use[i])
            # self.gen_models_att[i].to_gpu(GPU.gpus_to_use[i])
            self.dis_models[i].to_gpu(GPU.gpus_to_use[i])
            # self.dis_models_edges[i].to_gpu(GPU.gpus_to_use[i])
            self.det_models[i].to_gpu(GPU.gpus_to_use[i])
            # self.det_models_att[i].to_gpu(GPU.gpus_to_use[i])

    def save_models(self):
        xp = cuda.cupy
        test_cost = self.test_testset()
        if test_cost < self.last_best_result:
            # self.last_best_result = test_cost
            print '\nsaving the model with the test cost: ' + str(test_cost)

            serializers.save_hdf5('{0}enc.model'.format(self.save_dir), self.enc_models[0])
            serializers.save_hdf5('{0}enc.state'.format(self.save_dir), self.optimizer_enc)

            serializers.save_hdf5('{0}gen.model'.format(self.save_dir), self.gen_models[0])
            serializers.save_hdf5('{0}gen.state'.format(self.save_dir), self.optimizer_gen)

            # serializers.save_hdf5('{0}enc_att.model'.format(self.save_dir), self.enc_models_att[0])
            # serializers.save_hdf5('{0}enc_att.state'.format(self.save_dir), self.optimizer_enc_att)
            
            # serializers.save_hdf5('{0}gen_att.model'.format(self.save_dir), self.gen_models_att[0])
            # serializers.save_hdf5('{0}gen_att.state'.format(self.save_dir), self.optimizer_gen_att)

            serializers.save_hdf5('{0}dis.model'.format(self.save_dir), self.dis_models[0])
            serializers.save_hdf5('{0}dis.state'.format(self.save_dir), self.optimizer_dis)

            # serializers.save_hdf5('{0}dis_edges.model'.format(self.save_dir), self.dis_models_edges[0])
            # serializers.save_hdf5('{0}dis_edges.state'.format(self.save_dir), self.optimizer_dis_edges)

            serializers.save_hdf5('{0}det.model'.format(self.save_dir), self.det_models[0])
            serializers.save_hdf5('{0}det.state'.format(self.save_dir), self.optimizer_det)

            # serializers.save_hdf5('{0}det_att.model'.format(self.save_dir), self.det_models_att[0])
            # serializers.save_hdf5('{0}det_att.state'.format(self.save_dir), self.optimizer_det_att)

            sys.stdout.flush()
            # self.last_best_result = test_cost
            # print 'Saved a model with test error: ' + str(test_cost)

    def test_testset(self):
        xp = cuda.cupy
        test_rec_loss = 0
        num_batches = 50
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            for i in range(num_batches):
                att_images, images, _, _, _, _, _, _, _, _, _ , _, _, _, objects, obj_one_hot, describtions, descs_one_hot, _, _ = next(self.generator_test)
                att_images_multi, images_multi, _, _, objs_multi, objs_one_hot_multi, descs_multi, descs_one_hot_multi, _, _ = next(self.generator_test_multi)

                objs_multi = np.squeeze(objs_multi)
                objs_one_hot_multi = np.squeeze(objs_one_hot_multi)
                descs_multi = np.squeeze(descs_multi)
                descs_one_hot_multi = np.squeeze(descs_one_hot_multi)

                att_images = self.mix(att_images, att_images_multi)
                images = self.mix(images, images_multi)
                obj_one_hot = self.mix(obj_one_hot, objs_one_hot_multi)
                descs_one_hot = self.mix(descs_one_hot, descs_one_hot_multi)

                images = np.asarray(images, dtype=np.float32)
                images = images.transpose(1, 0, 2, 3, 4)
                att_images = np.asarray(att_images, dtype=np.float32)
                att_images = att_images.transpose(1, 0, 2, 3, 4)
                # for k, g in enumerate(GPU.gpus_to_use):
                cuda.get_device(GPU.main_gpu).use()
                
                img_input_batch_for_gpu = images[:, 0:self.batch_size // GPU.num_gpus]
                img_input_batch_for_gpu = np.reshape(img_input_batch_for_gpu,
                                                        (-1, self.num_channels, self.image_size, self.image_size))
                att_img_input_batch_for_gpu = att_images[:, 0:self.batch_size // GPU.num_gpus]
                att_img_input_batch_for_gpu = np.reshape(att_img_input_batch_for_gpu,
                                                        (-1, self.num_channels, self.image_size, self.image_size))
                obj_one_hot_batch_for_gpu = np.asarray(obj_one_hot[0:self.batch_size // GPU.num_gpus], dtype=np.float32)
                descs_one_hot_batch_for_gpu = np.asarray(descs_one_hot[0:self.batch_size // GPU.num_gpus], dtype=np.float32)

                objects = np.asarray(objects[0:self.batch_size // (2 * GPU.num_gpus)], dtype=np.float32)
                describtions = np.asarray(describtions[0:self.batch_size // (2 * GPU.num_gpus)], dtype=np.float32)
                objs_multi = np.asarray(objs_multi[0:self.batch_size // (2 * GPU.num_gpus)], dtype=np.float32)
                descs_multi = np.asarray(descs_multi[0:self.batch_size // (2 * GPU.num_gpus)], dtype=np.float32)
                objects = self.mix(objects, objs_multi)
                describtions = self.mix(describtions, descs_multi)

                # objects = np.asarray(objects[0:self.batch_size // ( GPU.num_gpus)], dtype=np.float32)
                # describtions = np.asarray(describtions[0:self.batch_size // (GPU.num_gpus)], dtype=np.float32)                
                objects = (objects / self.num_objects) - 1
                objects = objects[:, np.newaxis, np.newaxis, np.newaxis]
                objects = np.tile(objects, (1, 1, self.image_size, self.image_size))
                describtions = np.copy(describtions)
                describtions = (describtions / self.num_descs) - 1
                describtions = describtions[:, np.newaxis, np.newaxis, np.newaxis]
                describtions = np.tile(describtions, (1, 1, self.image_size, self.image_size))

                x_in = cuda.to_gpu(img_input_batch_for_gpu, GPU.main_gpu)
                obj_one_hot_batch_for_gpu = cuda.to_gpu(obj_one_hot_batch_for_gpu, GPU.main_gpu)
                descs_one_hot_batch_for_gpu = cuda.to_gpu(descs_one_hot_batch_for_gpu, GPU.main_gpu)
                objects = cuda.to_gpu(objects, GPU.main_gpu)
                describtions = cuda.to_gpu(describtions, GPU.main_gpu)
                x_in_att = cuda.to_gpu(att_img_input_batch_for_gpu, GPU.main_gpu)
                # gpu_batch_size = len(img_input_batch_for_gpu)

                z, _, _ = self.enc_models[0](Variable(x_in), Variable(obj_one_hot_batch_for_gpu), Variable(descs_one_hot_batch_for_gpu), train=False)
                data_att = self.gen_models[0](z, Variable(obj_one_hot_batch_for_gpu), Variable(descs_one_hot_batch_for_gpu), train=False)

                # z0, _, _ = self.enc_models_att[0](data_att, Variable(obj_one_hot_batch_for_gpu), Variable(descs_one_hot_batch_for_gpu), train=False)
                # data_whole = self.gen_models_att[0](z0, Variable(obj_one_hot_batch_for_gpu), Variable(descs_one_hot_batch_for_gpu), train=False)

                batch_rec_loss = F.mean_squared_error(data_att[:, :3].data, x_in_att) / 2
                batch_rec_loss += F.mean_squared_error(data_att[:, 3:].data, x_in) / 2
                # batch_rec_loss += F.mean_squared_error(data_whole[0::2].data, x_in[0::2])
                # batch_rec_loss += F.mean_squared_error(data_whole.data, x_in)
                # print batch_rec_loss.shape
                test_rec_loss += float(batch_rec_loss.data)

        test_rec_loss = test_rec_loss / num_batches
        print '\nrecent test cost: ' + str(test_rec_loss)
        return test_rec_loss

    def save_image(self, data, filename):
        image = ((data + 1) * 128).clip(0, 255).astype(np.uint8)
        image = image[:self.sample_image_rows * self.sample_image_cols]
        image = image.reshape(
            (self.sample_image_rows, self.sample_image_cols, data.shape[1], self.image_size,
                self.image_size)).transpose(
            (0, 3, 1, 4, 2)).reshape(
            (self.sample_image_rows * self.image_size, self.sample_image_cols * self.image_size, data.shape[1]))
        if data.shape[1] == 1:
            image = image.reshape(self.sample_image_rows * self.image_size,
                                    self.sample_image_cols * self.image_size)
        Image.fromarray(image).save(filename)

    def save_sample_images(self, epoch, batch):
        xp = cuda.cupy
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            images = self.sample_images
            images_att = self.sample_images_att
            images_crop = self.sample_images_crop
            cuda.get_device(GPU.main_gpu).use()
            objs_one_hot = cuda.to_gpu(self.sample_objs_one_hot, GPU.main_gpu)
            descs_one_hot = cuda.to_gpu(self.sample_descs_one_hot, GPU.main_gpu)
            objects_norm = cuda.to_gpu(self.sample_objects_norm, GPU.main_gpu)
            descs_norm = cuda.to_gpu(self.sample_describtions_norm, GPU.main_gpu)

            x_in = Variable(cuda.to_gpu(images, GPU.main_gpu))
            x_in_att = Variable(cuda.to_gpu(images_att, GPU.main_gpu))

            z, m, v = self.enc_models[0](x_in, Variable(objs_one_hot), Variable(descs_one_hot), train=False)
            data_att = self.gen_models[0](z, Variable(objs_one_hot), Variable(descs_one_hot), train=False)
            # z0, m, v = self.enc_models_att[0](Variable(data_att), Variable(objs_one_hot), Variable(descs_one_hot), train=False)
            # data_whole = self.gen_models_att[0](z0, Variable(objs_one_hot), Variable(descs_one_hot), train=False).data

            test_rec_loss = F.squared_difference(data_att[:, :3], xp.asarray(images_att))
            test_rec_loss = float(F.sum(test_rec_loss).data) / self.normer

            print '\ntest error on the random number of images: ' + str(test_rec_loss)
            self.save_image(cuda.to_cpu(data_att[:, :3].data), 'sample/att/{0:03d}_{1:07d}_att.png'.format(epoch, batch))
            self.save_image(cuda.to_cpu(data_att[:, 3:].data), 'sample/whole/{0:03d}_{1:07d}_whole.png'.format(epoch, batch))
            self.save_image(cuda.to_cpu(self.conv_2d_kernel(data_att[:, 3:], GPU.main_gpu).data), 'sample/e_whole/{0:03d}_{1:07d}_whole_edges.png'.format(epoch, batch))
            self.save_image(cuda.to_cpu(self.conv_2d_kernel(data_att[:, :3], GPU.main_gpu).data), 'sample/e_att/{0:03d}_{1:07d}_att_edges.png'.format(epoch, batch))

            if batch == 0:
                self.save_image(cuda.to_cpu(self.conv_2d_kernel(x_in, GPU.main_gpu).data), 'sample/e_whole/org_pic_edges.png')
                self.save_image(cuda.to_cpu(self.conv_2d_kernel(x_in_att, GPU.main_gpu).data), 'sample/e_att/org_att_edges.png')
                self.save_image(images, 'sample/whole/org_pic.png')
                self.save_image(images_att, 'sample/att/org_att.png')

    """
        Load the saved model and optimizer
    """

    def load_model(self):
        try:
            file_path = os.path.join(os.path.dirname(__file__), self.save_dir)

            serializers.load_hdf5(file_path + 'enc.model', self.enc_model)
            #serializers.load_hdf5(file_path + 'enc.state', self.optimizer_enc)

            serializers.load_hdf5(file_path + 'gen.model', self.gen_model)
            #serializers.load_hdf5(file_path + 'gen.state', self.optimizer_gen)

            # serializers.load_hdf5(file_path + 'enc_att.model', self.enc_model_att)
            # #serializers.load_hdf5(file_path + 'enc_att.state', self.optimizer_enc_att)
            
            # serializers.load_hdf5(file_path + 'gen_att.model', self.gen_model_att)
            # #serializers.load_hdf5(file_path + 'gen_att.state', self.optimizer_gen_att)

            serializers.load_hdf5(file_path + 'dis.model', self.dis_model)
            #serializers.load_hdf5(file_path + 'dis.state', self.optimizer_dis)

            # serializers.load_hdf5(file_path + 'dis_edges.model', self.dis_model)
            #serializers.load_hdf5(file_path + 'dis.state', self.optimizer_dis)

            serializers.load_hdf5(file_path + 'det.model', self.det_model)
            #serializers.load_hdf5(file_path + 'det.state', self.optimizer_det)

            # serializers.load_hdf5(file_path + 'det_att.model', self.det_model_att)
            # #serializers.load_hdf5(file_path + 'det_att.state', self.optimizer_det_att)


        except Exception as inst:
            print inst
            print 'cannot load model from {}'.format(file_path)


        self.enc_models = [self.enc_model]
        self.gen_models = [self.gen_model]
        # self.enc_models_att = [self.enc_model_att]
        # self.gen_models_att = [self.gen_model_att]
        # self.gen_mask_models = [self.gen_mask_model]
        self.dis_models = [self.dis_model]
        # self.dis_models_edges = [self.dis_model_edges]
        self.det_models = [self.det_model]
        # self.det_models_att = [self.det_model_att]     

        for i in range(GPU.num_gpus - 1):
            self.enc_models.append(copy.deepcopy(self.enc_model))
            self.gen_models.append(copy.deepcopy(self.gen_model))
            # self.enc_models_att.append(copy.deepcopy(self.enc_model_att))
            # self.gen_models_att.append(copy.deepcopy(self.gen_model_att))
            # self.gen_mask_models.append(copy.deepcopy(self.gen_mask_model))
            self.dis_models.append(copy.deepcopy(self.dis_model))
            # self.dis_models_edges.append(copy.deepcopy(self.dis_model_edges))
            self.det_models.append(copy.deepcopy(self.det_model))
            # self.det_models_att.append(copy.deepcopy(self.det_model_att))

