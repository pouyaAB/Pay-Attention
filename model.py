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
import autoencoders.tower
from nf_mdn_rnn import RobotController
from DatasetController_morph import DatasetController

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
                 sample_image_cols=5, sample_image_rows=2, load_models=True):

        self.dataset_ctrl = DatasetController(batch_size=int(batch_size), sequence_input=sequence_size, sequence_output=0, read_jpgs=True)
        self.num_tasks = len(config['tasks'])
        self.image_size = image_size
        self.normer = image_size * image_size * 3 * 60
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

        self.num_descs = self.dataset_ctrl.num_all_objects_describtors + 1
        self.num_objects = self.dataset_ctrl.num_all_objects + 1

        images_att, images, _, _, _ , _, _, _, objects, objs_one_hot, descriptions, descs_one_hot = next(self.generator_test)

        images = images[:, 0]
        images_att = images_att[:, 0]
        images = np.reshape(images, (-1, self.num_channels, self.image_size, self.image_size))
        images_att = np.reshape(images_att, (-1, self.num_channels, self.image_size, self.image_size))

        # Preparing a set of sample images to test the network on and save the reconstructions etc
        # Take a look at the function save_sample_images
        sample_size = self.sample_image_cols * self.sample_image_rows
        self.sample_images = np.asarray(images[:sample_size], dtype=np.float32)
        self.sample_images_att = np.asarray(images_att[:sample_size], dtype=np.float32)
        self.sample_objs_one_hot = np.asarray(objs_one_hot[:sample_size], dtype=np.float32)
        self.sample_descs_one_hot = np.asarray(descs_one_hot[:sample_size], dtype=np.float32)

        objects = np.asarray(objects[:sample_size], dtype=np.float32)
        descriptions = np.asarray(descriptions[:sample_size], dtype=np.float32)

        # target objects code ... take a look at the dictionary in the DatasetController_morph class
        print('objects invloved code:')
        print np.squeeze(objects)[:sample_size]
        print np.squeeze(descriptions)[:sample_size]

        self.generator = self.dataset_ctrl.get_next_batch(task=['5002', '5001'], from_start_prob=0, human=False, sth_sth=False, camera='camera-1')

        self.enc_model = autoencoders.tower.Encoder_text_tower(density=8, size=image_size, latent_size=latent_size, channel=self.num_channels, num_objects=self.num_objects - 1, num_descriptions=self.num_descs - 1)
        self.gen_model = autoencoders.tower.Generator_text(density=8, size=image_size, latent_size=latent_size, channel=self.num_channels * 2, num_objects=self.num_objects - 1, num_descriptions=self.num_descs - 1)
        self.dis_model = autoencoders.tower.Discriminator_texual(density=8, size=image_size, channel=self.num_channels,
                                                        num_objects=self.num_objects, num_descriptions=self.num_descs)

        self.enc_models = [self.enc_model]
        self.gen_models = [self.gen_model]
        self.dis_models = [self.dis_model]

        self.learning_rate = 0.0001
        self.WeightDecay = 0.00001

        self.optimizer_enc = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        self.optimizer_enc.setup(self.enc_models[0])
        self.optimizer_enc.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        self.optimizer_gen = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        self.optimizer_gen.setup(self.gen_models[0])
        self.optimizer_gen.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        self.optimizer_dis = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        self.optimizer_dis.setup(self.dis_models[0])
        self.optimizer_dis.add_hook(chainer.optimizer.WeightDecay(self.WeightDecay))

        for i in range(GPU.num_gpus - 1):
            self.enc_models.append(copy.deepcopy(self.enc_model))
            self.gen_models.append(copy.deepcopy(self.gen_model))
            self.dis_models.append(copy.deepcopy(self.dis_model))

        self.batch_gpu_threads = [None] * GPU.num_gpus

        if self.load_models:
            self.load_model()
        self.to_gpu()

    def reset_all(self, models):
        for model in models:
            model.reset_state()

    def clear_grads(self, models):
        for model in models:
            model.cleargrads()

    def show_image(self, images):
        img = images
        img = img.transpose(1, 2, 0)
        img = (img + 1) * 127.5
        img = img.astype(np.uint8)
        print img.dtype, np.max(img), np.min(img), np.shape(img)
        img = Image.fromarray(img, "RGB")
        img.show()

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
                att_images, images, _, \
                _, _, \
                joints, _, batch_one_hot,\
                objs, objs_one_hot, descs, descs_one_hot = next(self.generator)

                att_images = np.concatenate((att_images, np.copy(images)), axis=2)

                # Calling the function which will manage the batch to be passed to GPU's
                for k, g in enumerate(GPU.gpus_to_use):
                    self.batch_gpu_threads[k] = threading.Thread(target=self.handle_gpu_batch, args=(batches_passed, 
                                                                                                    batch_start_time, 
                                                                                                    k, g, 
                                                                                                    images, att_images, 
                                                                                                    joints, 
                                                                                                    batch_one_hot, 
                                                                                                    objs, objs_one_hot, 
                                                                                                    descs, descs_one_hot))
                    self.batch_gpu_threads[k].start()

                for i in range(GPU.num_gpus):
                    self.batch_gpu_threads[i].join()
                self.add_grads()

                if self.train_autoencoder:
                    self.optimizer_enc.update()
                    self.optimizer_gen.update()

                if self.train_dis:
                    self.optimizer_dis.update()

                self.copy_params()

                current_batch = batches_passed
                if current_batch % self.save_sample_image_interval == 0:
                    self.save_sample_images(epoch=epoch, batch=current_batch)

                if current_batch % self.save_model_period == self.save_model_period - 1:
                    self.save_models()

            self.save_models()
            self.save_sample_images(epoch=epoch, batch=batches_passed)

    def handle_gpu_batch(self, 
                        batches_passed, 
                        batch_start_time, 
                        k, g, # GPU info
                        images, # images from the dataset
                        att_images, # masked frames from teacher network
                        joints, # commands
                        batch_one_hot, # task_id
                        objects, objs_one_hot, # target object shape
                        descriptions, descriptions_one_hot): # target object description
                        
        xp = cuda.cupy
        cuda.get_device(g).use()
        self.dis_models[k].cleargrads()
        self.gen_models[k].cleargrads()
        self.enc_models[k].cleargrads()

        gpu_batch_size = self.batch_size // GPU.num_gpus

        # Preparing the inputs
        images = images.transpose(1, 0, 2, 3, 4)
        images = images[:, k * gpu_batch_size:(k + 1) * gpu_batch_size]
        att_images = att_images.transpose(1, 0, 2, 3, 4)
        att_images = att_images[:, k * gpu_batch_size:(k + 1) * gpu_batch_size]

        objects = np.asarray(objects[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.int32)
        objects = np.repeat(objects[np.newaxis], self.sequence_size, axis=0)
        objects = np.squeeze(np.reshape(objects, (self.sequence_size * gpu_batch_size, -1)))

        descriptions = np.asarray(descriptions[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.int32)
        descriptions = np.repeat(descriptions[np.newaxis], self.sequence_size, axis=0)
        descriptions = np.squeeze(np.reshape(descriptions, (self.sequence_size * gpu_batch_size, -1)))

        descriptions_norm = np.asarray(descriptions, dtype=np.float32)
        real_objects = objects - 1
        real_descriptions = descriptions - 1

        objects_one_hot = np.asarray(objs_one_hot[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        objects_one_hot = np.repeat(objects_one_hot[np.newaxis], self.sequence_size, axis=0)
        objects_one_hot = np.reshape(objects_one_hot, (self.sequence_size * gpu_batch_size, -1))

        descs_one_hot = np.asarray(descriptions_one_hot[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        descs_one_hot = np.repeat(descs_one_hot[np.newaxis], self.sequence_size, axis=0)
        descs_one_hot = np.reshape(descs_one_hot, (self.sequence_size * gpu_batch_size, -1))

        images = np.reshape(images, (-1, self.num_channels, self.image_size, self.image_size))
        x_in = cuda.to_gpu(np.asarray(images, dtype=np.float32), g)
        att_images = np.reshape(att_images, (-1, self.num_channels * 2, self.image_size, self.image_size))
        x_in_att = cuda.to_gpu(np.asarray(att_images, dtype=np.float32), g)

        joints = joints.transpose(1, 0, 2)
        joints = np.asarray(joints[:, k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        joints = Variable(cuda.to_gpu(joints, g))

        batch_one_hot = np.asarray(batch_one_hot[k * gpu_batch_size:(k + 1) * gpu_batch_size], dtype=np.float32)
        batch_one_hot = Variable(cuda.to_gpu(batch_one_hot, g))
        objects_var = Variable(cuda.to_gpu(objects, g))
        desc_var = Variable(cuda.to_gpu(descriptions, g))

        real_objects_var = Variable(cuda.to_gpu(real_objects, g))
        real_desc_var = Variable(cuda.to_gpu(real_descriptions, g))

        gpu_images_size = len(images)

        # Passing the inputs to the Encoder
        z0, mean, var = self.enc_models[k](Variable(x_in), Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)) , train=self.train_autoencoder)    
        l_prior = F.gaussian_kl_divergence(mean, var) / (2 * self.normer)
        
        # Passing the inputs to the Generator
        # The first 3 channels of the output is the masked frame 
        # and the last 3 channels are the original frame reconstruction
        x0_att = self.gen_models[k](z0, Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)), train=self.train_autoencoder)
        reconstruction_loss_att = F.mean_squared_error(x0_att[:, :3], x_in_att[:, :3])

        # The original frame's reconstruction is looped back into the Encoder and Generator
        z00, mean0, var0 = self.enc_models[k](x0_att[:, 3:], Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)) , train=self.train_autoencoder)    
        l_prior += F.gaussian_kl_divergence(mean0, var0) / (2 * self.normer)

        x0_att0 = self.gen_models[k](z00, Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)), train=self.train_autoencoder)
        reconstruction_loss_att += F.mean_squared_error(x0_att0, x0_att)

        # Discriminator trying to classify the Generator's outputs
        y0_att, d0_att, l0_att = self.dis_models[k](x0_att[:, :3], train=self.train_autoencoder)
        y0_att0, d0_att0, l0_att0 = self.dis_models[k](x0_att0[:, :3], train=self.train_autoencoder)

        y0_whole, d0_whole, l0_whole = self.dis_models[k](x0_att[:, 3:], att=False, train=self.train_autoencoder)
        y0_whole0, d0_whole0, l0_whole0 = self.dis_models[k](x0_att0[:, 3:], att=False, train=self.train_autoencoder)

        # Discriminator classification error
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

        # Passing a random vector sampled from Normal Distribution to the generator 
        z1 = Variable(cuda.to_gpu(xp.random.normal(0, 1, (gpu_images_size, self.latent_size), dtype=np.float32), g))
        x1_att = self.gen_models[k](z1, Variable(cuda.to_gpu(objects_one_hot, g)), Variable(cuda.to_gpu(descs_one_hot, g)), train=self.train_autoencoder)
        y1_att, d1_att, _ = self.dis_models[k](x1_att[:, :3], train=self.train_dis)
        y1_whole, d1_whole, _ = self.dis_models[k](x1_att[:, 3:], att=False, train=self.train_dis)

        l_dis_obj_fake_att = F.softmax_cross_entropy(y1_att, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g))) 
        l_dis_desc_fake_att = F.softmax_cross_entropy(d1_att, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))
        l_dis_obj_fake_whole = F.softmax_cross_entropy(y1_whole, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g))) 
        l_dis_desc_fake_whole = F.softmax_cross_entropy(d1_whole, Variable(cuda.to_gpu(xp.zeros(gpu_images_size).astype(np.int32), g)))
        l_dis_random_fake = (l_dis_obj_fake_att + l_dis_desc_fake_att + l_dis_obj_fake_whole + l_dis_desc_fake_whole) / (4 * gpu_images_size)

        #Discriminator receiving the real frames from the teacher network and the dataset
        y2_att, d2_att, l2_att = self.dis_models[k](x_in_att[:, :3], train=self.train_dis)
        y2_whole, d2_whole, l2_whole = self.dis_models[k](x_in_att[:, 3:], att=False, train=self.train_dis)

        l_dis_obj_real_att = F.softmax_cross_entropy(y2_att, objects_var) 
        l_dis_desc_real_att = F.softmax_cross_entropy(d2_att, desc_var)
        l_dis_obj_real_whole = F.softmax_cross_entropy(y2_whole, objects_var) 
        l_dis_desc_real_whole = F.softmax_cross_entropy(d2_whole, desc_var)
        l_dis_real = (l_dis_obj_real_att + l_dis_desc_real_att + l_dis_obj_real_whole + l_dis_desc_real_whole) / (4 * gpu_images_size)

        # Losses
        l_feature_similarity_att = F.mean_squared_error(l0_att, l2_att) + F.mean_squared_error(l0_att0, l0_att)
        l_feature_similarity_whole = F.mean_squared_error(l0_whole, l2_whole)  + F.mean_squared_error(l0_whole0, l0_whole)
        l_feature_similarity = (l_feature_similarity_att + l_feature_similarity_whole)
        reconstruction_loss = reconstruction_loss_att
        l_dis_sum = (l_dis_random_fake + l_dis_0 + l_dis_real)
        loss_enc = l_prior + l_feature_similarity
        loss_dis = l_dis_sum
        loss_gen = l_gen_0 + l_feature_similarity + reconstruction_loss - l_dis_sum

        self.train_dis = True
        self.train_autoencoder = True

        if self.train_autoencoder:
            self.enc_models[k].cleargrads()
            self.gen_models[k].cleargrads()
            loss_net = loss_enc + loss_gen
            loss_net.backward()

        x0_att.unchain()
        x1_att.unchain()

        if self.train_dis:
            self.dis_models[k].cleargrads()
            loss_dis.backward()

        sys.stdout.write('\r' + str(batches_passed) + '/' + str(1000) +
                         ' time: {0:0.2f}, enc:{1:0.4f}, gen:{2:0.4f}, dis:{3:0.4f}, rec_att:{4:0.4f}, rec:{5:0.4f}, pri:{6:0.4f}, fea:{7:0.4f} ,det_loss:{8:0.4f}, lstm:{9:0.4f}'.format(
                             time.time() - batch_start_time,
                             float(loss_enc.data),
                             float(loss_gen.data),
                             float(loss_dis.data),
                             float(reconstruction_loss.data),
                             float(0.0),
                             float(l_prior.data),
                             float(l_feature_similarity.data),
                             float(0.0),
                             float(0.0)
                         ))
        sys.stdout.flush()  # important

    def copy_params(self):
        for i in range(1, GPU.num_gpus):
            self.enc_models[i].copyparams(self.enc_models[0])
            self.gen_models[i].copyparams(self.gen_models[0])
            self.dis_models[i].copyparams(self.dis_models[0])

    def add_grads(self):
        for j in range(1, GPU.num_gpus):
            self.enc_models[0].addgrads(self.enc_models[j])
            self.gen_models[0].addgrads(self.gen_models[j])
            self.dis_models[0].addgrads(self.dis_models[j])

    def to_gpu(self):
        for i in range(GPU.num_gpus):
            self.enc_models[i].to_gpu(GPU.gpus_to_use[i])
            self.gen_models[i].to_gpu(GPU.gpus_to_use[i])
            self.dis_models[i].to_gpu(GPU.gpus_to_use[i])

    def save_models(self):
        xp = cuda.cupy
        test_cost = self.test_testset()
        if test_cost < self.last_best_result:
            print '\nsaving the model with the test cost: ' + str(test_cost)

            serializers.save_hdf5('{0}enc.model'.format(self.save_dir), self.enc_models[0])
            serializers.save_hdf5('{0}enc.state'.format(self.save_dir), self.optimizer_enc)

            serializers.save_hdf5('{0}gen.model'.format(self.save_dir), self.gen_models[0])
            serializers.save_hdf5('{0}gen.state'.format(self.save_dir), self.optimizer_gen)

            serializers.save_hdf5('{0}dis.model'.format(self.save_dir), self.dis_models[0])
            serializers.save_hdf5('{0}dis.state'.format(self.save_dir), self.optimizer_dis)

            serializers.save_hdf5('{0}rnn_mdn_adverserial_e2e.model'.format(self.save_dir), self.mdn_models[0])
            serializers.save_hdf5('{0}rnn_mdn_adverserial_e2e.state'.format(self.save_dir), self.optimizer_mdn)

            sys.stdout.flush()

    def test_testset(self):
        xp = cuda.cupy
        test_rec_loss = 0
        test_lstm_loss = 0
        num_batches = 50
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            for i in range(num_batches):
                att_images, images,\
                 _, _, _, \
                joints, _, batch_one_hot,\
                 _, obj_one_hot,\
                _, descs_one_hot = next(self.generator_test)

                images = np.asarray(images, dtype=np.float32)
                images = images.transpose(1, 0, 2, 3, 4)
                att_images = np.asarray(att_images, dtype=np.float32)
                att_images = att_images.transpose(1, 0, 2, 3, 4)
                cuda.get_device(GPU.main_gpu).use()
                
                img_input_batch_for_gpu = images[:, 0:self.batch_size // GPU.num_gpus]
                img_input_batch_for_gpu = np.reshape(img_input_batch_for_gpu,
                                                        (-1, self.num_channels, self.image_size, self.image_size))
                att_img_input_batch_for_gpu = att_images[:, 0:self.batch_size // GPU.num_gpus]
                att_img_input_batch_for_gpu = np.reshape(att_img_input_batch_for_gpu,
                                                        (-1, self.num_channels, self.image_size, self.image_size))
                obj_one_hot_batch_for_gpu = np.asarray(obj_one_hot[0:self.batch_size // GPU.num_gpus], dtype=np.float32)
                obj_one_hot_batch_for_gpu = np.repeat(obj_one_hot_batch_for_gpu[np.newaxis], self.sequence_size, axis=0)
                obj_one_hot_batch_for_gpu = np.reshape(obj_one_hot_batch_for_gpu, (-1, self.num_objects - 1))
                
                descs_one_hot_batch_for_gpu = np.asarray(descs_one_hot[0:self.batch_size // GPU.num_gpus], dtype=np.float32)
                descs_one_hot_batch_for_gpu = np.repeat(descs_one_hot_batch_for_gpu[np.newaxis], self.sequence_size, axis=0)
                descs_one_hot_batch_for_gpu = np.reshape(descs_one_hot_batch_for_gpu, (-1, self.num_descs - 1))

                joints = joints.transpose(1, 0, 2)
                joints = np.asarray(joints[:, 0:self.batch_size // GPU.num_gpus], dtype=np.float32)
                joints = Variable(cuda.to_gpu(joints, GPU.main_gpu))

                batch_one_hot = np.asarray(batch_one_hot[0:self.batch_size // GPU.num_gpus], dtype=np.float32)
                batch_one_hot = Variable(cuda.to_gpu(batch_one_hot, GPU.main_gpu))

                x_in = cuda.to_gpu(img_input_batch_for_gpu, GPU.main_gpu)
                obj_one_hot_batch_for_gpu_xp = cuda.to_gpu(obj_one_hot_batch_for_gpu, GPU.main_gpu)
                descs_one_hot_batch_for_gpu_xp = cuda.to_gpu(descs_one_hot_batch_for_gpu, GPU.main_gpu)
                x_in_att = cuda.to_gpu(att_img_input_batch_for_gpu, GPU.main_gpu)

                z, _, _ = self.enc_models[0](Variable(x_in), Variable(obj_one_hot_batch_for_gpu_xp), Variable(descs_one_hot_batch_for_gpu_xp), train=False)
                data_att = self.gen_models[0](z, Variable(obj_one_hot_batch_for_gpu_xp), Variable(descs_one_hot_batch_for_gpu_xp), train=False)

                z0_seq = F.reshape(z, (self.sequence_size, self.batch_size// GPU.num_gpus, self.latent_size))
                seq_objects_one_hot = np.reshape(obj_one_hot_batch_for_gpu, (self.sequence_size, self.batch_size// GPU.num_gpus, -1))
                seq_descs_one_hot = np.reshape(descs_one_hot_batch_for_gpu, (self.sequence_size, self.batch_size// GPU.num_gpus, -1))
                task_encoding = F.concat((batch_one_hot, Variable(cuda.to_gpu(seq_objects_one_hot[0], GPU.main_gpu)), Variable(cuda.to_gpu(seq_descs_one_hot[0], GPU.main_gpu))), axis=-1)

                batch_rec_loss = F.mean_squared_error(data_att[:, :3].data, x_in_att) / 2
                batch_rec_loss += F.mean_squared_error(data_att[:, 3:].data, x_in) / 2
                test_rec_loss += float(batch_rec_loss.data)

        test_rec_loss = test_rec_loss / num_batches
        print '\nrecent test cost: ' + str(test_rec_loss)
        return test_rec_loss

    def save_image(self, data, filename):
        image = ((data + 1) * 128).clip(0, 255).astype(np.uint8)
        image = image[:self.sample_image_rows * self.sample_image_cols]
        image = image.reshape(
            (self.sample_image_rows, self.sample_image_cols, self.num_channels, self.image_size,
                self.image_size)).transpose(
            (0, 3, 1, 4, 2)).reshape(
            (self.sample_image_rows * self.image_size, self.sample_image_cols * self.image_size, self.num_channels))
        if self.num_channels == 1:
            image = image.reshape(self.sample_image_rows * self.image_size,
                                    self.sample_image_cols * self.image_size)
        Image.fromarray(image).save(filename)

    def save_sample_images(self, epoch, batch):

        xp = cuda.cupy
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

            images = self.sample_images
            images_att = self.sample_images_att
            cuda.get_device(GPU.main_gpu).use()
            objs_one_hot = cuda.to_gpu(self.sample_objs_one_hot, GPU.main_gpu)
            descs_one_hot = cuda.to_gpu(self.sample_descs_one_hot, GPU.main_gpu)

            z, m, v = self.enc_models[0](Variable(cuda.to_gpu(images, GPU.main_gpu)), Variable(objs_one_hot), Variable(descs_one_hot), train=False)
            data_att = self.gen_models[0](z, Variable(objs_one_hot), Variable(descs_one_hot), train=False).data

            test_rec_loss = F.squared_difference(data_att[:, :3], xp.asarray(images_att))
            test_rec_loss = float(F.sum(test_rec_loss).data) / self.normer

            print '\ntest error on the random number of images: ' + str(test_rec_loss)
            self.save_image(cuda.to_cpu(data_att[:, :3]), 'sample/{0:03d}_{1:07d}_att.png'.format(epoch, batch))
            self.save_image(cuda.to_cpu(data_att[:, 3:]), 'sample/{0:03d}_{1:07d}_whole.png'.format(epoch, batch))
            if batch == 0:
                self.save_image(images, 'sample/org_pic.png')
                self.save_image(images_att, 'sample/org_att.png')

    """
        Load the saved model and optimizer
    """

    def load_model(self):
        try:
            file_path = os.path.join(os.path.dirname(__file__), self.save_dir)

            serializers.load_hdf5(file_path + 'enc.model', self.enc_model)
            serializers.load_hdf5(file_path + 'enc.state', self.optimizer_enc)

            serializers.load_hdf5(file_path + 'gen.model', self.gen_model)
            serializers.load_hdf5(file_path + 'gen.state', self.optimizer_gen)

            serializers.load_hdf5(file_path + 'dis.model', self.dis_model)
            serializers.load_hdf5(file_path + 'dis.state', self.optimizer_dis)

        except Exception as inst:
            print inst
            print 'cannot load model from {}'.format(file_path)


        self.enc_models = [self.enc_model]
        self.gen_models = [self.gen_model]
        self.dis_models = [self.dis_model]

        for i in range(GPU.num_gpus - 1):
            self.enc_models.append(copy.deepcopy(self.enc_model))
            self.gen_models.append(copy.deepcopy(self.gen_model))
            self.dis_models.append(copy.deepcopy(self.dis_model))

