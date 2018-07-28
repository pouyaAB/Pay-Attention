import numpy as np
import time
import sys
import csv
from local_config import config
import scipy.ndimage

locals().update(config)

sys.path.append("/home/d3gan/catkin_ws/src/ros_teleoprtate/al5d/scripts/")

from test_network_robot import TestNetworkRobot
import os
import copy
import chainer.functions as F
from PIL import Image
import threading
import signal
import copy

from sklearn.decomposition import PCA
from sklearn.externals import joblib

import tensorflow as tf
from gpu import GPU
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
import autoencoders.tower
# from nf_mdn_rnn import MDN_RNN
from nf_mdn_rnn import MDN_RNN
from Utilities import Utils
import matplotlib.pyplot as plt
import cv2

from DatasetController_morph import DatasetController


def signal_handler(signal, frame):
    global MT
    MT.dataset_ctrl.end_thread = True
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class ModelTester:
    """
        Handles the model interactions
    """

    def __init__(self, image_size, latent_size, output_size=7, num_channels=3, task='5002', tasks=['5001', '5002', '5003', '5004'], save_dir="model/"):
        self.image_size = image_size
        self.latent_size = latent_size
        self.num_channels = num_channels
        self.save_dir = save_dir
        self.output_size = output_size
        self.batch_size = 1
        self.task = task
        self.tasks = tasks
        self.attention_latent_size = 32
        self.attention_size = 28
        self.auto_regressive = False
        self.use_all_cameras = False
        self.use_all_cameras_stacked_on_channel = False
        if self.use_all_cameras_stacked_on_channel:
            self.num_channels = self.num_channels * 3
        self.which_camera_to_use = 1

        self.dataset_ctrl = TestNetworkRobot(self.image_size, config['record_path'], config['robot_command_file'],
                                             config['camera_topics'], cache_size=1,
                                             cameras_switch=[False, True, False])

        # while not all(self.dataset_ctrl.all_caches_filled):
        #     print 'Waiting for the caches to fill!'
        #     time.sleep(1)
        
        DC = DatasetController(batch_size=32, sequence_input=1, sequence_output=0, read_jpgs=True)
        self.g = DC.get_next_batch(sth_sth=False, task=['5002', '5001'], from_start=False, train=True, camera='camera-' + str(self.which_camera_to_use))

        self.num_descs = DC.num_all_objects_describtors
        self.num_objects = DC.num_all_objects

        self.enc_model = autoencoder.Encoder_text_tower(density=8, size=self.image_size, latent_size=self.latent_size,
                                             channel=self.num_channels, num_objects=self.num_objects, num_describtions=self.num_descs)
        self.gen_model = autoencoder.Generator_text(density=8, size=image_size, latent_size=latent_size, channel=self.num_channels * 2, 
                                                    num_objects=self.num_objects, num_describtions=self.num_descs)
        self.mdn_model = MDN_RNN((latent_size) + self.num_descs + self.num_objects, hidden_dimension, output_size, num_mixture, auto_regressive=self.auto_regressive)

        self.dataset_path = '/home/d3gan/development/datasets/record/sth_sth_224'
        self.source_dataset_path = dataset_path
        self.bag_of_words = {}
        self.annotations = {}
        self.reverse_annotations = {}
        self.string_size = 10
        self.vocabularySize = 32
        self.annotated_tasks = [5001, 5002]
        self.bag_of_words = self.fill_bag_of_words(self.annotated_tasks)
        for task in self.annotated_tasks:
            self.read_annotations(task)

        # raw_sentence = "push white plate from left to right"
        raw_sentence = "pick-up black dumble"
        self.sentence = self.encode_annotation(task, raw_sentence)

        for i, label in enumerate(self.sentence):
            if int(label) in DC.objects:
                self.which_object = DC.objects[int(label)]
            if int(label) in DC.objects_describtors:
                self.which_describtor = DC.objects_describtors[int(label)]
        
        self.objects_norm = np.zeros((1), dtype=np.float32)
        self.objects_norm[0] = (float(self.which_object) / (self.num_objects + 1)) - 1
        self.objects_norm = self.objects_norm[:, np.newaxis, np.newaxis, np.newaxis]
        self.objects_norm = np.tile(self.objects_norm, (1, 1, self.image_size, self.image_size))
        self.describtions_norm = np.zeros((1), dtype=np.float32)
        self.describtions_norm[0] = (float(self.which_describtor) / (self.num_descs + 1)) - 1
        self.describtions_norm = self.describtions_norm[:, np.newaxis, np.newaxis, np.newaxis]
        self.describtions_norm = np.tile(self.describtions_norm, (1, 1, self.image_size, self.image_size))

        self.object_involved_one_hot = np.zeros((1, self.num_objects))
        self.describtors_involved_one_hot = np.zeros((1, self.num_descs))
        self.object_involved_one_hot[0, self.which_object - 1] = 1 
        self.describtors_involved_one_hot[0, self.which_describtor - 1] = 1
        self.object_involved_one_hot = np.asarray(self.object_involved_one_hot, dtype=np.float32)
        self.describtors_involved_one_hot = np.asarray(self.describtors_involved_one_hot, dtype=np.float32)

        self.load_model()
        self.to_gpu()
        self.real_time_test()

    def get_task_one_hot_vector(self, task):
        one_hot = np.zeros((self.batch_size, len(self.tasks)), dtype=np.float32)
        for i in range(self.batch_size):
            one_hot[i][int(task) - 5001] = 1

        return one_hot

    def encode_annotation(self, task, sentence):
        words = sentence.split()
        encoded = np.zeros((self.string_size), dtype=int)
        for i, word in enumerate(words):
            encoded[i] = int(self.bag_of_words[word])
        
        return encoded

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

    def real_time_test(self):
        predicted = np.zeros((self.output_size), dtype=np.float32)
        self.mdn_model.reset_state()
        while True:
            input_online_images, _ = self.dataset_ctrl.get_next_batch(batch_size=1, camera_id=self.which_camera_to_use, channel_first=True, homography=True)
            # resized_image = Image.fromarray(np.uint8((input_online_images[0] + 1) * 127.5).transpose((1, 2, 0))).resize((self.image_size, self.image_size), resample=Image.ANTIALIAS)
            # # toShow.show()
            # # raw_input()

            # resized_image = resized_image.convert('RGB')
            # resized_image = np.asarray(resized_image, dtype=np.float32)
            # resized_image = resized_image.transpose((2, 0, 1))
            # resized_image = resized_image / 127.5 - 1
            # resized_image = cv2.resize(((input_online_images[0] + 1) * 127.5).transpose(1, 2, 0), dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            # resized_image = ((resized_image / 127.5) -1).transpose(2, 0, 1)
            # self.show_image(resized_image[np.newaxis, :])
            # att_latent = self.pca_model.transform(att.reshape(1, -1) - self.pca_data_mean)
            # inv_att = self.pca_model.inverse_transform(att_latent) + self.pca_data_mean
            # inv_att_0 = np.reshape(inv_att, (28, 28))
            # plt.imshow(inv_att_0, cmap='gray')
            # plt.show()
            # att = np.asarray(att[np.newaxis, np.newaxis, :], dtype=np.float32)
            # att_latent = np.asarray(att_latent[np.newaxis, :], dtype=np.float32)
            # att_latent = np.swapaxes(att_latent, 0, 1)
            # att_0 = np.reshape(att, (28, 28))
            # plt.imshow(att_0, cmap='gray')
            # plt.show()
            # self.show_image(input_online_images)
            # input_online_images = input_online_images.transpose(0, 3, 2, 1)
            # input_images, _, _, _, _, _, _, _, _, _, _, _, _, _ = next(self.g)
            # input_images = np.asarray(input_images, dtype=np.float32)
            # input_images = np.squeeze(input_images, axis=1)
            # self.show_image(input_images[np.newaxis])
            # toshow = resized_image.transpose(1, 2, 0)
            # toshow = Image.fromarray(np.uint8((toshow + 1) * 127.5))
            # toshow.show()
            input_online_images = np.asarray(input_online_images[0], dtype=np.float32)
            # input_images[0] = resized_image
            x_in = cuda.to_gpu(input_online_images[np.newaxis], GPU.main_gpu)
            z0, mean, var = self.enc_model(Variable(x_in),Variable(cuda.to_gpu(self.object_involved_one_hot, GPU.main_gpu)), Variable(cuda.to_gpu(self.describtors_involved_one_hot, GPU.main_gpu)), train=False)
            
            input_online_images = np.transpose(input_online_images, (1, 2, 0))
            tshow = cv2.cvtColor((input_online_images + 1) * 127.5, cv2.COLOR_BGR2RGB)
            cv2.imshow('Input image original', cv2.resize(tshow.astype(np.uint8), (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(10)
            # x_in_train = cuda.to_gpu(input_images, GPU.main_gpu)
            # z0_train, _, _ = self.enc_model(Variable(x_in_train), train=False)

            x_in_att = self.gen_model(z0, Variable(cuda.to_gpu(self.object_involved_one_hot, GPU.main_gpu)), Variable(cuda.to_gpu(self.describtors_involved_one_hot, GPU.main_gpu)), train=False)

            tshow = (cuda.to_cpu(x_in_att[:, :3].data)[0] + 1) * 127.5
            tshow = cv2.cvtColor(tshow.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
            cv2.imshow('Reconstruction att image', cv2.resize(tshow.astype(np.uint8), (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            cv2.waitKey(1)

            tshow = (cuda.to_cpu(x_in_att[:, 3:].data)[0] + 1) * 127.5
            tshow = cv2.cvtColor(tshow.transpose((1, 2, 0)), cv2.COLOR_BGR2RGB)
            cv2.imshow('Reconstruction whole image', cv2.resize(tshow.astype(np.uint8), (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            cv2.waitKey(1)

            # self.show_image(cuda.to_cpu(x_in.data))
            
            task_one_hot = self.get_task_one_hot_vector(self.task)
            task_one_hot = Variable(cuda.to_gpu(task_one_hot, GPU.main_gpu))
            input_feature = F.concat((z0,  Variable(cuda.to_gpu(self.object_involved_one_hot, GPU.main_gpu)), Variable(cuda.to_gpu(self.describtors_involved_one_hot, GPU.main_gpu))), axis=-1)
            input_feature = F.expand_dims(input_feature, axis=0)
            dummy_joints = Variable(cuda.to_gpu(np.zeros((1, 1, self.output_size), dtype=np.float32), GPU.main_gpu))

            if self.auto_regressive:
                _, sample = self.mdn_model(data_in=task_one_hot, z=input_feature, data_out=dummy_joints, return_sample=True, train=False)
                predicted = sample
            else:
                _, sample = self.mdn_model(data_in=task_one_hot, z=input_feature, data_out=dummy_joints, return_sample=True, train=False)
                # sample = self.mdn_model(data_in=task_one_hot, z=input_feature, train=False)
                predicted = cuda.to_cpu(sample.data)[0]

            # self.write_command(sample.data[0])
            self.dataset_ctrl.send_command(predicted)
            print predicted
            # raw_input()
            # print input_online_images.shape

    def morph_attention_and_image(self, images, attentions):
        toRet_att = np.empty((1, self.image_size, self.image_size))

        att = np.asarray(attentions, dtype=np.float32)
        mins = np.min(att, axis=1)
        maxs = np.max(att, axis=1)
        att -= mins[:, np.newaxis]
        att = (2 * att) / maxs[:, np.newaxis] - 1
        att = np.reshape(att, (-1, 28, 28))
        att = scipy.ndimage.zoom(att, (1, 28 / 28.0, 28 / 28.0), order=3)
        att = att[0]
        att = F.max_pooling_2d(Variable(att[np.newaxis, np.newaxis, :]), 6).data
        att[att < 0.95] = 0
        # plt.imshow(att[0][0], cmap='gray')
        # plt.show()
        # image = images.transpose((1, 2, 0))
        # image = np.transpose(images, (1,2,0))
        att = scipy.ndimage.zoom(att[0,0], (self.image_size / float(att.shape[2]), self.image_size / float(att.shape[2])), order=2)
        # toshow = scipy.ndimage.zoom(att, (1, 1), order=1)
        # cv2.imshow('att', toshow)
        # cv2.waitKey(1)

        # att = F.unpooling_2d(Variable(att), 4, cover_all=True).data
        # cv2.imshow('image', att*255)
        # cv2.waitKey(1000)
        images = images * att[:, :, np.newaxis]
        # images = image.transpose((2, 0, 1))
        # images = np.transpose(image, (2, 0, 1))
        toRet_att = att[np.newaxis, :, :]
        
        # im = Image.fromarray(np.uint8(((image + 1) * 127.5)))
        # im.show()
        
        return images, toRet_att

    def show_image(self, images):
        for i in range(images.shape[0]):
            # moving axis to use plt: i.e [4,100,100] to [100,100,4]
            img = images[i]
            img = img.transpose(1, 2, 0)
            img = (img +1) *127.5
            img = img.astype(np.uint8)
            print img.dtype, np.max(img), np.min(img), np.shape(img)
            img = Image.fromarray(img, "RGB")
            img.show()
            # raw_input()

    def write_command(self, command):
        with open(os.path.join(record_path, 'commands.csv'), 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(command)

    def load_model(self):
        try:
            file_path = os.path.join(os.path.dirname(__file__), self.save_dir)

            serializers.load_hdf5(file_path + 'enc.model', self.enc_model)
            # serializers.load_hdf5(file_path + 'enc_adverserial.state', self.optimizer_enc)

            # serializers.load_hdf5(file_path + 'dis.model', self.dis_model)
            # serializers.load_hdf5(file_path + 'dis.state', self.optimizer_dis)
            #
            serializers.load_hdf5(file_path + 'gen.model', self.gen_model)
            # serializers.load_hdf5(file_path + 'gen.state', self.optimizer_gen)
            #
            # # serializers.load_hdf5se_path + 'gen_mask.state', self.optimizer_gen)
            #
            # if self.ae_mode == 'AAE':
            #     serializers.load_hdf5(file_path + 'aae.model', self.aae_dis_model)
            #     serializers.load_hdf5(file_path + 'aae.state', self.optimizer_aae_dis)
            serializers.load_hdf5(file_path + 'rnn_mdn_adverserial.model', self.mdn_model)
            # serializers.load_hdf5(file_path + 'rnn_mdn.state', self.optimizer_mdn)
            print('Models has been loaded!')
        except Exception as inst:
            print inst
            print 'cannot load model from {}'.format(file_path)
            sys.exit(0)

    def to_gpu(self):
        self.enc_model.to_gpu(GPU.main_gpu)
        self.gen_model.to_gpu(GPU.main_gpu)
        self.mdn_model.to_gpu(GPU.main_gpu)

if __name__ == '__main__':
    global MT
    MT = ModelTester(image_size, latent_size, task=task, tasks=tasks, output_size=7)

    # time.sleep(4)
    # images, joints = TNR.get_next_batch(4, 1)
    # print(np.shape(images), np.shape(joints))
    # rospy.spin()