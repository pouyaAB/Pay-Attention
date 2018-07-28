import numpy as np
import time
import sys
import os
import copy
import chainer.functions as F
import signal
import random
from PIL import Image

from gpu import GPU
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
import autoencoders.tower
import matplotlib.pyplot as plt
from local_config import config
from autoencoder import Attention
# from nf_rnn_vanilla import MDN_RNN
from nf_mdn_rnn import MDN_RNN
import threading
import csv

def signal_handler(signal, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
locals().update(config)


class TrainLstm:
    def __init__(self):

        self.include_sth_sth = False
        self.use_all_cameras_stack_on_channel = False
        self.use_transformed_images = False
        self.source_dataset_path = dataset_path
        self.dest_dataset_path = self.source_dataset_path
        self.model_path = "model/"
        self.image_size = image_size
        self.latent_size = latent_size
        self.out_filepath = './processed_inputs_' + str(image_size) + '_latent_'+ str(latent_size) + '_top_path_cycled_e2e/'
        self.batch_size = 100
        self.sequence_size = 15
        self.string_size = 10
        self.hidden_dim = hidden_dimension
        self.cameras = ['camera-0', 'camera-1', 'camera-2']
        self.num_mixture = num_mixture
        self.dataset_path = dataset_path
        self.num_channels = num_channels
        self.num_epoch = 10000
        self.max_sequence_len = {}
        self.save_model_period = 300
        self.csv_col_num = csv_col_num
        self.output_size = output_size
        self.best_test_result = float('Inf')
        self.save_dir = "model/"
        self.dataset = {}
        self.tasks = tasks
        self.train_percent = 0.8
        self.train_dis = True
        self.train_mdn_only = True
        self.train_lstm = False
        self.tasks_train_on = ['5001', '5002']

        self.best_test_loss = 100
        self.step_size = 2
        self.learning_rate = 0.001

        print 'best_test_loss: ' + str(self.best_test_loss)
        print 'learning_rate: ' + str(self.learning_rate)
        print 'sequence_size: ' + str(self.sequence_size)
        print 'step_size: ' + str(self.step_size)

        self.output_size = 7

        self.annotations = {}
        self.reverse_annotations = {}
        self.annotated_tasks = [5001, 5002]
        self.bag_of_words = self.fill_bag_of_words(self.annotated_tasks)
        for task in self.annotated_tasks:
            self.read_annotations(task)

        self.objects_describtors = {"white" : 1,
                        "blue": 2,
                        "black-white": 3,
                        "black": 4,
                        "red": 5}
        self.objects = {"plate": 1,
                        "box": 2,
                        "qr-box": 3,
                        "bubble-wrap": 4,
                        "bowl": 5,
                        "towel": 6,
                        "dumble": 7,
                        "ring": 8}
        self.objects_describtors = {self.bag_of_words[x]:y for x,y in self.objects_describtors.iteritems()}
        self.objects = {self.bag_of_words[x]:y for x,y in self.objects.iteritems()}
        self.num_all_objects = len(self.objects.keys())
        self.num_all_objects_describtors = len(self.objects_describtors.keys())

        self.joints_std = {
                    5001: [0.0, 0.1859, 0.0752, 0.0862, 0.0814, 0.2842, 0.0],
                    5002: [0.0, 0.2066, 0.0874, 0.0942, 0.0658, 0.2066, 0.0],
                    5003: [0.0, 0.1723, 0.0741, 0.0936, 0.0651, 0.1722, 0.0],
                    5004: [0.0, 0.1879, 0.0731, 0.0813, 0.0756, 0.3407, 0.0]
                }

        # self.mdn_model = MDN_RNN(self.latent_size, self.hidden_dim, self.output_size)
        self.mdn_model = MDN_RNN(self.latent_size + self.num_all_objects + self.num_all_objects_describtors, hidden_dimension, output_size, num_mixture, auto_regressive=False)
        # self.dis_model = autoencoder.seq_Discriminator(in_dim=self.latent_size + self.output_size + 4, hidden_dim=64)

        self.mdn_models = [self.mdn_model]
        # self.dis_models = [self.dis_model]

        for _ in range(GPU.num_gpus - 1):
            self.mdn_models.append(copy.deepcopy(self.mdn_model))
            # self.dis_models.append(copy.deepcopy(self.dis_model))
        
        self.optimizer_mdn = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        # self.optimizer_mdn = optimizers.RMSpropGraves(lr=self.learning_rate)
        self.optimizer_mdn.setup(self.mdn_models[0])
        self.optimizer_mdn.add_hook(chainer.optimizer.WeightDecay(0.00001))

        # self.optimizer_dis = optimizers.Adam(alpha=self.learning_rate, beta1=0.9)
        # self.optimizer_dis.setup(self.dis_models[0])
        # self.optimizer_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))

        self.batch_gpu_threads = [None] * GPU.num_gpus
        self.load_model()
        self.to_gpu()
        self.load_dataset()

    def read_annotations(self, task):
        self.reverse_annotations[task] = {}
        with open(os.path.join(self.source_dataset_path, str(task) + '_task_annotation.csv'), 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                words = row[0].split()
                key = ''
                for word in words:
                    key += str(self.bag_of_words[word]) + ' '
                key += '0'
                self.annotations[key] = row[1:]
                for dem in row[1:]:
                    self.reverse_annotations[task][dem] = key

    def fill_bag_of_words(self, tasks):
        unique_words = []
        max_len = 0
        bag = {}
        for task in tasks:
            with open(os.path.join(self.source_dataset_path, str(task) + '_task_annotation.csv'), 'rb') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in spamreader:
                    words = row[0].split()
                    if len(words) > max_len:
                        max_len = len(words)
                    for word in words:
                        if word not in unique_words:
                            unique_words.append(word)

        for i, word in enumerate(unique_words):
            bag[word] = i + 1

        if max_len + 1 > self.string_size:
            print("ERROR: provided string size is smaller than the biggest annotation!")

        return bag

    def load_dataset(self):
        for dir_name in os.listdir(self.out_filepath):
            if os.path.isdir(os.path.join(self.out_filepath, dir_name)):
                print('loading from directory: %s' % dir_name)
                self.dataset[dir_name] = {}
                max_len = self.process_tasks(dir_name)
                self.max_sequence_len[int(dir_name)] = max_len

    def get_attention_label(self, task, dem_index):
        correct_sentence = self.reverse_annotations[int(task)][dem_index]
        
        labels = correct_sentence.split()
        key_toRet = np.zeros((self.string_size), dtype=int)
        which_describtor = 0
        which_object = 0
        for i, label in enumerate(labels):
            key_toRet[i] = int(label)

            if which_object == 0 and int(label) in self.objects:
                which_object = self.objects[int(label)]
            if which_describtor == 0 and int(label) in self.objects_describtors:
                which_describtor = self.objects_describtors[int(label)]

        return key_toRet, which_object, which_describtor

    def process_tasks(self, dir_name):
        source = os.path.join(self.out_filepath, dir_name)

        max_len = 0
        for subdir_name in sorted(os.listdir(source)):

            if subdir_name in self.reverse_annotations[int(dir_name)].keys():
                dir = os.path.join(source, subdir_name)
                joints = np.load(os.path.join(source, str(subdir_name) + '-joints.npy'))
                if self.use_all_cameras_stack_on_channel:
                    images_cameras = np.load(os.path.join(dir, 'cameras.npy'))
                else:
                    # images_camera_0 = np.load(os.path.join(dir, 'camera-0.npy'))
                    images_camera_1 = np.load(os.path.join(dir, 'camera-1.npy'))
                    # images_camera_2 = np.load(os.path.join(dir, 'camera-2.npy'))
                for i in range(self.step_size):
                    indexes = range(i, joints.shape[0], self.step_size)
                    if self.use_all_cameras_stack_on_channel:
                        images_cameras_skipped = images_cameras[indexes]
                    else:
                        # images_skipped_0 = images_camera_0[indexes]
                        images_skipped_1 = images_camera_1[indexes]
                        # images_skipped_2 = images_camera_2[indexes]
                    joints_skipped = joints[indexes]
                    # images_skipped, joints_skipped = self.repeat_first_last(images_skipped, joints_skipped, num_repeats=5)
                    if max_len < joints_skipped.shape[0]:
                        max_len = joints_skipped.shape[0]

                    if joints_skipped.shape[0] > self.sequence_size:
                        if subdir_name + '/' + str(i) not in self.dataset[dir_name].keys():
                            self.dataset[dir_name][subdir_name + '/' + str(i)] = {}
                        if self.use_all_cameras_stack_on_channel:
                            self.dataset[dir_name][subdir_name + '/' + str(i)]['cameras'] = np.asarray(images_cameras_skipped, dtype=np.float32)
                        else:
                            # toShow= Image.fromarray(np.uint8((images_skipped_1[0] + 1) * 127.5))
                            # toShow.show()

                            # self.dataset[dir_name][subdir_name + '/' + str(i)]['camera-0'] = np.asarray(images_skipped_0, dtype=np.float32)
                            self.dataset[dir_name][subdir_name + '/' + str(i)]['camera-1'] = np.asarray(images_skipped_1, dtype=np.float32)
                            # self.dataset[dir_name][subdir_name + '/' + str(i)]['camera-2'] = np.asarray(images_skipped_2, dtype=np.float32)
                        self.dataset[dir_name][subdir_name + '/' + str(i)]['joints'] = np.asarray(joints_skipped, dtype=np.float32)
                    else:
                        print "too small"
        
        return max_len

    def get_task_one_hot_vector(self, joints):
        one_hot = np.zeros((self.batch_size, len(self.tasks)), dtype=np.float32)
        for i in range(self.batch_size):
            one_hot[i][int(joints[i][0][1]) - 5001] = 1

        return one_hot

    def apply_noise(self, task, joints):
        eps = np.absolute(np.random.normal(0, 0.1, joints.shape))

        task_std = self.joints_std[int(task)]
        task_std = np.broadcast_to(task_std, joints.shape)

        final_std = np.multiply(task_std, eps)

        return np.asarray(np.random.normal(loc=joints, scale=final_std), dtype=np.float32)

    def repeat_first_last(self, images, joints, num_repeats=5):
        first_image = np.expand_dims(images[0], axis=0)
        first_image = np.repeat(first_image, num_repeats, axis=0)
        images = np.concatenate((first_image, images), axis=0)

        last_image = np.expand_dims(images[-1], axis=0)
        last_image = np.repeat(last_image, num_repeats, axis=0)
        images = np.concatenate((images, last_image), axis=0)

        first_joint = np.expand_dims(joints[0], axis=0)
        first_joint = np.repeat(first_joint, num_repeats, axis=0)
        joints = np.concatenate((first_joint, joints), axis=0)

        last_joint = np.expand_dims(joints[-1], axis=0)
        last_joint = np.repeat(last_joint, num_repeats, axis=0)
        joints = np.concatenate((joints, last_joint), axis=0)

        return images, joints

    def get_random_latents(self, req_task=None, train=True, from_start_prob = 0.1, use_all_cameras=False):
        batch_images = np.empty((self.batch_size, self.sequence_size, self.latent_size), dtype=np.float32)
        batch_joints = np.empty((self.batch_size, self.sequence_size, self.csv_col_num), dtype=np.float32)
        object_involved_one_hot = np.zeros((self.batch_size, self.num_all_objects))
        describtors_involved_one_hot = np.zeros((self.batch_size, self.num_all_objects_describtors))

        for i in range(self.batch_size):
            if type(req_task) == list:
                rand_task = np.random.randint(len(req_task), size=1)[0]
                chosen_task = req_task[rand_task]
            else:
                chosen_task = req_task
            task = chosen_task
            if task is None:
                task_index = np.random.randint(len(self.dataset.keys()))
                task = self.dataset.keys()[task_index]

            if train:
                train_dems = self.dataset[str(task)].keys()
                train_dems = train_dems[:int(len(train_dems) * self.train_percent)]
                dem_index = np.random.randint(len(train_dems))
                dem = train_dems[dem_index]
            else:
                test_dems = self.dataset[str(task)].keys()
                test_dems = test_dems[int(len(test_dems) * self.train_percent):]
                dem_index = np.random.randint(len(test_dems))
                dem = test_dems[dem_index]
            

            images = self.dataset[task][dem]['camera-1']
            if use_all_cameras:
                which_camera = np.random.randint(len(self.cameras), size=1)[0]
                images = self.dataset[task][dem]['camera-' + str(which_camera)]
            joints = self.dataset[task][dem]['joints']

            # start_index = np.random.randint(np.shape(images)[0] - 2 * self.sequence_size) #because of the repeating
            start_index = np.random.randint(np.shape(images)[0] - self.sequence_size)
            coin_toss = random.uniform(0, 1)
            if coin_toss < from_start_prob:
                start_index = 0

            batch_images[i] = images[start_index : start_index + self.sequence_size]
            batch_joints[i] = joints[start_index :  start_index + self.sequence_size]
        
            _, which_object, which_describtor = self.get_attention_label(task, dem[:-2])

            object_involved_one_hot[i, which_object - 1] = 1
            describtors_involved_one_hot[i, which_describtor - 1] = 1

        batch_one_hot = self.get_task_one_hot_vector(batch_joints)
        noisy_joints = self.apply_noise(task, batch_joints[:, :, 3:])
        # noisy_joints = np.concatenate((np.expand_dims(noisy_joints[:, : , 0], axis=2), np.expand_dims(noisy_joints[:, :, -1], axis=2)), axis=-1)
        return batch_images, batch_joints[:, :, 3:], batch_one_hot, object_involved_one_hot, describtors_involved_one_hot

    def train(self):

        for epoch in range(self.num_epoch):
            print '\n ------------- epoch {0} started ------------'.format(epoch)
            for batch_start in range(0, 1000):
                latents, joints, one_hot, objs_one_hot, descs_one_hot = self.get_random_latents(req_task=self.tasks_train_on)
                # latents1, joints1, one_hot1, _, _ = self.get_random_latents(req_task=self.tasks_train_on)

                batch_start_time = time.time()
                for k, g in enumerate(GPU.gpus_to_use):
                    self.gpu_train(k, g, batch_start_time, batch_start, latents, joints, one_hot, objs_one_hot, descs_one_hot)
                #      self.batch_gpu_threads[k] = threading.Thread(target=self.gpu_train, 
                #         args=(k, g, batch_start_time, batch_start, latents, joints, latents1, joints1, one_hot, one_hot1))
                #      self.batch_gpu_threads[k].start()
                
                # for i in range(GPU.num_gpus):
                #     self.batch_gpu_threads[i].join()
                # self.add_grads()
                # if self.train_lstm:
                #     self.optimizer_mdn.update()
                # if self.train_dis:
                #     self.optimizer_dis.update()
                # self.copy_params()

                current_batch = batch_start

                if current_batch % self.save_model_period == self.save_model_period - 1:
                    # self.optimizer_mdn.new_epoch()
                    self.save_models()

    def gpu_train(self, k, g, batch_start_time, batch_start, latents, joints, one_hot, objs_one_hot, descs_one_hot , train=True):
        with chainer.using_config('train', train), chainer.using_config('enable_backprop', train):
            xp = cuda.cupy
            cuda.get_device(g).use()
            self.reset_all([self.mdn_models[k]])
            self.mdn_models[k].cleargrads()
            # self.dis_models[k].cleargrads()
            gpu_batch_size = latents.shape[0]

            objs_one_hot = np.asarray(np.repeat(objs_one_hot[:, np.newaxis], latents.shape[1], axis=1), dtype=np.float32)
            descs_one_hot = np.asarray(np.repeat(descs_one_hot[:, np.newaxis], latents.shape[1], axis=1), dtype=np.float32)
            latents = np.concatenate((latents, objs_one_hot, descs_one_hot), axis=-1)

            latents = np.swapaxes(latents, 0, 1)
            joints = np.swapaxes(joints, 0, 1)

            latents = cuda.to_gpu(latents, g)
            joints = cuda.to_gpu(joints, g)
            batch_one_hot = cuda.to_gpu(one_hot, g)

            #rnn part

            mdn_loss, predicted_joints = self.mdn_models[k](data_in=batch_one_hot, z=latents[:-1], data_out=joints[1:], return_sample=True, train=train)

            if not train:
                return mdn_loss

            # latents1 = cuda.to_gpu(latents1, g)
            # joints1 = cuda.to_gpu(joints1, g)

            # random_latents = Variable(cuda.to_gpu(xp.random.normal(0, 1, (self.sequence_size, gpu_batch_size, self.latent_size), dtype=np.float32), g))
            # _, predicted_joints_fake = self.mdn_models[k](data_in=batch_one_hot, z=random_latents[:-1], data_out=joints[1:], return_sample=True)

            # predicted_joints = F.swapaxes(predicted_joints, 0, 1)
            # predicted_joints_fake = F.swapaxes(predicted_joints_fake, 0, 1)
            # random_latents = F.swapaxes(random_latents, 0, 1)
            # latents = F.swapaxes(latents, 0, 1)
            # joints = np.swapaxes(joints, 0, 1)

            # fake_random_sequence = F.concat((predicted_joints_fake, random_latents[:, :-1]), axis=-1)
            # real_sequence = F.concat((joints1[:, 1:], latents1[:, :-1]), axis=-1)
            # fake_sequence = F.concat((predicted_joints, latents[:, :-1]), axis=-1)
            # fake_sequence_real = F.concat((joints[:, 1:], latents[:, :-1]), axis=-1)

            # y0, _ = self.dis_models[k](real_sequence, batch_one_hot1, train=train)
            # _, l1_real = self.dis_models[k](fake_sequence_real, batch_one_hot, train=train)
            # y1, l1 = self.dis_models[k](fake_sequence, batch_one_hot, train=train)
            # y2, _ = self.dis_models[k](fake_random_sequence, batch_one_hot, train=train)

            # l_dis_real = F.softmax_cross_entropy(y0, Variable(cuda.to_gpu(xp.ones(gpu_batch_size).astype(np.int32), g))) / gpu_batch_size
            # l_dis_fake = F.softmax_cross_entropy(y1, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size).astype(np.int32), g))) / gpu_batch_size
            # l_dis_fake_random = F.softmax_cross_entropy(y2, Variable(cuda.to_gpu(xp.zeros(gpu_batch_size).astype(np.int32), g))) /gpu_batch_size

            # l_feature_similarity = 100 * F.mean_squared_error(l1_real, l1)

            # lstm_cost *=1000
            # loss_dis = (l_dis_real + l_dis_fake + l_dis_fake_random)
            # if self.train_mdn_only:
            #     lstm_cost = mdn_loss
            # else:
            #     lstm_cost = l_feature_similarity - loss_dis

            # self.train_lstm = True
            # self.train_dis = True
            lstm_cost = mdn_loss
            if train:
                self.mdn_models[k].cleargrads()
                lstm_cost.backward()
                self.optimizer_mdn.update()
            # fake_random_sequence.unchain()
            # fake_sequence.unchain()

            # self.dis_models[k].cleargrads()
            # loss_dis.backward()
            # self.optimizer_dis.update()

                sys.stdout.write('\r' + str(batch_start) + '/' + str(1000) +
                                        ' time: {0:0.2f} lstm:{1:0.4f}, dis:{2:0.4f}, mdn:{3:0.4f} similarity:{4:0.4f} train_dis:{5}'.format(
                                            time.time() - batch_start_time, 
                                            float(lstm_cost.data), 
                                            float(0.0),
                                            float(mdn_loss.data),
                                            float(0.0),
                                            self.train_dis
                                            ))
                sys.stdout.flush()  # important

            return mdn_loss
    
    def test(self):
        test_loss = 0
        num_batches = 50
        for batch_start in range(0, num_batches):
            latents, joints, one_hot, objs_one_hot, descs_one_hot = self.get_random_latents(req_task=self.tasks_train_on, train=False)
            batch_start_time = time.time()

            test_loss += self.gpu_train(0, GPU.main_gpu, batch_start_time, batch_start, latents, joints, one_hot, objs_one_hot, descs_one_hot, train=False)

        return test_loss.data / num_batches

    def reset_all(self, models):
        for model in models:
            model.reset_state()

    def save_models(self):
        loss = self.test()
        if loss < self.best_test_loss:
            print('\nNew test loss: ' + str(loss))
            self.best_test_loss = loss

            serializers.save_hdf5('{0}rnn_mdn_adverserial.model'.format(self.save_dir), self.mdn_models[0])
            serializers.save_hdf5('{0}rnn_mdn_adverserial.state'.format(self.save_dir), self.optimizer_mdn)

            print '\nmodel saved!'
            sys.stdout.flush()

        else:
            serializers.save_hdf5('{0}rnn_mdn_adverserial_overfit.model'.format(self.save_dir), self.mdn_models[0])
            serializers.save_hdf5('{0}rnn_mdn_adverserial_overfit.state'.format(self.save_dir), self.optimizer_mdn)

            print('\ntest loss: ' + str(loss) + ' best loss: ' + str(self.best_test_loss))

    def load_model(self):
        try:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model_path)

            serializers.load_hdf5(file_path + 'rnn_mdn_adverserial.model', self.mdn_model)
            # serializers.load_hdf5(file_path + 'rnn_mdn_adverserial.state', self.optimizer_mdn)

            self.mdn_models = [self.mdn_model]
            for _ in range(GPU.num_gpus - 1):
                self.mdn_models.append(copy.deepcopy(self.mdn_model))

        except Exception as inst:
            print inst
            print 'cannot load the encoder model from {}'.format(file_path)

    def to_gpu(self):
        for i in range(GPU.num_gpus):
            self.mdn_models[i].to_gpu(GPU.gpus_to_use[i])
            # self.dis_models[i].to_gpu(GPU.gpus_to_use[i])

    def copy_params(self):
        for i in range(1, GPU.num_gpus):
            self.mdn_models[i].copyparams(self.mdn_models[0])
            # self.dis_models[i].copyparams(self.dis_models[0])

    def add_grads(self):
        for j in range(1, GPU.num_gpus):
            self.mdn_models[0].addgrads(self.mdn_models[j])
            # self.dis_models[0].addgrads(self.dis_models[j])


if __name__ == '__main__':
    TL = TrainLstm()
    TL.train()
