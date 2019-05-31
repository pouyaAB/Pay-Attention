import numpy as np
import pandas as pd
import time
import os
from PIL import Image
import csv
import random
import scipy.ndimage
import chainer.functions as F
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from image_transformer import imageTransformer
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from skimage import transform

from local_config import config

locals().update(config)


def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


class DatasetController:
    def __init__(self, batch_size, sequence_input, sequence_output, string_size=10, read_jpgs=False, shuffle_att=False):
        self.read_jpgs = read_jpgs
        self.tasks = tasks
        self.batch_size = batch_size
        self.sequence_input = sequence_input
        self.sequence_output = sequence_output
        self.sequence_size = sequence_input + sequence_output
        self.num_channels = num_channels
        self.image_size = image_size
        self.csv_col_num = csv_col_num
        self.shuffle_att = shuffle_att
        # self.debug = True
        self.cameras = cameras
        self.attention_size = 28
        self.train_percentage = 0.8

        self.transformer = imageTransformer("empty")
        self.source_dataset_path = dataset_path
        if read_jpgs:
            self.dest_dataset_path = self.source_dataset_path
        else:
            self.dest_dataset_path = dataset_path + '_compressed'
        
        self.out_filepath_attention = './processed_inputs_attention_28_new_objects/'

        self.source_sth_sth_path = sth_sth_path
        if read_jpgs:
            self.dest_sth_sth_path = self.source_sth_sth_path
        else:
            self.dest_sth_sth_path = sth_sth_path + '_compressed'
        
        self.joints_std = { 5001: [0.0, 0.1859, 0.0752, 0.0862, 0.0814, 0.2842, 0.0],
                            5002: [0.0, 0.2066, 0.0874, 0.0942, 0.0658, 0.2066, 0.0],
                            5003: [0.0, 0.1723, 0.0741, 0.0936, 0.0651, 0.1722, 0.0],
                            5004: [0.0, 0.1879, 0.0731, 0.0813, 0.0756, 0.3407, 0.0]}

        self.annotations = {}
        self.reverse_annotations = {}
        self.string_size = string_size
        self.annotated_tasks = [5001, 5002]
        self.bag_of_words = self.fill_bag_of_words(self.annotated_tasks)
        for task in self.annotated_tasks:
            self.read_annotations(task)

        if not read_jpgs:
            create_dir(self.dest_dataset_path)
            create_dir(self.dest_sth_sth_path)
        self.step_size = 2

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

        if not read_jpgs and not os.path.exists(os.path.join(self.dest_sth_sth_path, 'complete')):
            self.process_sth_sth()
            open(os.path.join(self.dest_sth_sth_path, 'complete'), 'w+').close()
        if not read_jpgs and not os.path.exists(os.path.join(self.dest_dataset_path, 'complete')):
            self.process_dataset()
            open(os.path.join(self.dest_dataset_path, 'complete'), 'w+').close()
        # self.get_random_demonstration()

        self.train_folders = {}
        self.test_folders = {}
        self.separate_train_train()

    def separate_train_train(self):
        for task_id in self.annotated_tasks:
            # task_path = os.path.join(self.dest_dataset_path, task_id)
            # folders = [name for name in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, name))]

            folders = self.reverse_annotations[int(task_id)].keys()
            random.shuffle(folders)
            num_folders = len(folders)

            self.train_folders[str(task_id)] = folders[:int(num_folders * self.train_percentage)]
            self.test_folders[str(task_id)] = folders[int(num_folders * self.train_percentage):]

    def encode_annotation(self, sentence):
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

    def get_random_demonstration(self, task_id, train=True):
        # print len(self.tasks.keys()), self.tasks.keys()[rand_task]
        if task_id is None:
            rand_task = np.random.randint(len(self.tasks), size=1)[0]
            task_id = self.tasks[rand_task]
        elif type(task_id) == list:
            rand_task = np.random.randint(len(task_id), size=1)[0]
            task_id = task_id[rand_task]
        if train:
            folders =  self.train_folders[task_id]
        else:
            folders =  self.test_folders[task_id]

        num_dems = len(folders)
        rand_dem = np.random.randint(num_dems, size=1)[0]

        return task_id, folders[rand_dem]

    def get_task_one_hot_vector(self, joints):
        one_hot = np.zeros((self.batch_size, len(self.tasks)), dtype=np.float32)
        for i in range(self.batch_size):
            one_hot[i][int(joints[i][0][1]) - 5001] = 1

        return one_hot

    def get_attention_label(self, task, dem_index):
        correct_sentence = self.reverse_annotations[int(task)][dem_index]

        labels = correct_sentence.split()
        key_toRet = np.zeros((self.string_size), dtype=int)
        which_describtor = 0
        which_object = 0
        wrong_which_describtor = 0
        wrong_which_object = 0
        for i, label in enumerate(labels):
            key_toRet[i] = int(label)
            
            if which_object == 0 and int(label) in self.objects:
                wrong_labels = [item for item in self.objects.keys() if item not in [int(label)]]
                random_wrong_key = np.random.randint(len(wrong_labels), size=1)[0]
                wrong_which_object = self.objects[int(wrong_labels[random_wrong_key])]
                which_object = self.objects[int(label)]
            if which_describtor == 0 and int(label) in self.objects_describtors:
                wrong_labels = [item for item in self.objects_describtors.keys() if item not in [int(label)]]
                random_wrong_key = np.random.randint(len(wrong_labels), size=1)[0]
                wrong_which_describtor = self.objects_describtors[int(wrong_labels[random_wrong_key])]
                which_describtor = self.objects_describtors[int(label)]

        return key_toRet, which_object, which_describtor, wrong_which_object, wrong_which_describtor

    def apply_noise(self, task, joints):
        eps = np.absolute(np.random.normal(0, 0.1, joints.shape))

        task_std = self.joints_std[int(task)]
        task_std = np.broadcast_to(task_std, joints.shape)

        final_std = np.multiply(task_std, eps)

        return np.asarray(np.random.normal(loc=joints, scale=final_std), dtype=np.float32)

    def extract_att_rect(self, att, image):
        att_h = att.shape[-1]
        image_h = image.shape[0]
        att_loc = np.argmax(att)
        rect_c_x = att_loc % att_h
        rect_c_y = att_loc // att_h
        offset = (image_h / 4)
        rect_x0 = rect_c_x - offset
        rect_x1 = rect_c_x + offset

        rect_y0 = rect_c_y - offset
        rect_y1 = rect_c_y + offset

        if rect_x0 < 0:
            rect_x0 = 0
            rect_x1 = 2 * offset
        if rect_x1 > image_h:
            rect_x1 = image_h
            rect_x0 = image_h - 2 * offset
        
        if rect_y0 < 0:
            rect_y0 = 0
            rect_y1 = 2 * offset
        if rect_y1 > image_h:
            rect_y1 = image_h
            rect_y0 = image_h - 2 * offset
        
        return image[rect_y0: rect_y1, rect_x0 : rect_x1]


    def morph_attention_and_image(self, images, attentions):
        toRet_att = np.empty((images.shape[0], self.sequence_size, 1, self.image_size, self.image_size))
        toRet_cropped_att = np.empty((images.shape[0], self.sequence_size, 3, self.image_size, self.image_size))

        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                att_orig = np.reshape(attentions[j], (28, 28))
                att = F.max_pooling_2d(Variable(att_orig[np.newaxis, np.newaxis, :]), 7).data
                att[att < 0.95] = 0
                if self.shuffle_att:
                    att[:] = 0
                    rand_w, rand_h= np.random.randint(att.shape[-1], size=2)
                    att[0,0, rand_w, rand_h] = 1
                # print np.sum(att)
                # plt.imshow(att[0][0], cmap='gray')
                # plt.show()
                image = images[i][j].transpose((1, 2, 0))
                att = scipy.ndimage.zoom(att[0,0], (self.image_size / float(att.shape[2]), self.image_size / float(att.shape[2])), order=1)
                image = image * att[:, :, np.newaxis]
                cropped_image = self.extract_att_rect(att, image)
                # cropped_image = image[att_loc_x * scaler: (att_loc_x + 1) * scaler, att_loc_y * scaler: (att_loc_y + 1) * scaler]
                
                cropped_image = transform.resize(cropped_image, (self.image_size, self.image_size))
                images[i, j] = image.transpose((2, 0, 1))
                toRet_cropped_att[i, j] = cropped_image.transpose((2, 0, 1))
                # im = Image.fromarray(np.uint8(((cropped_image + 1) * 127.5)))
                # im.show()
                # im = Image.fromarray(np.uint8(((image + 1) * 127.5)))
                # im.show()
        
        return images, toRet_att, toRet_cropped_att

    def get_next_batch(self, task=None, sth_sth=False, human=False, joint_history=0, channel_first=True, from_start=False, from_start_prob=0.1, train=True, camera='camera-1', use_vgg=False):

        while True:
            human_images = np.empty(
                (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            human_paths = []
            original_robot_images = np.empty(
                (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            robot_images = np.empty(
                (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            robot_images_secondary = np.empty(
                (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            robot_attentions = np.empty(
                (self.batch_size, self.sequence_size, self.attention_size, self.attention_size))
            toRet_robot_attentions = np.empty(
                (self.batch_size, self.sequence_size, 1, self.image_size, self.image_size))
            toRet_robot_cropped_attentions = np.empty(
                (self.batch_size, self.sequence_size, 3, self.image_size, self.image_size))
            robot_paths = []
            sth_sth_images = np.empty(
                (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            sth_sth_paths = []
            batch_joints = np.empty((self.batch_size, self.sequence_size + joint_history, self.csv_col_num))

            attention_sentence = np.zeros((self.batch_size, self.string_size))
            object_involved = np.zeros((self.batch_size))
            describtors_involved = np.zeros((self.batch_size))

            object_involved_one_hot = np.zeros((self.batch_size, self.num_all_objects))
            describtors_involved_one_hot = np.zeros((self.batch_size, self.num_all_objects_describtors))
            wrong_object_involved_one_hot = np.zeros((self.batch_size, self.num_all_objects))
            wrong_describtors_involved_one_hot = np.zeros((self.batch_size, self.num_all_objects_describtors))
            human_paths = []
            robot_paths = []
            sth_sth_paths = []
            for i in range(self.batch_size):

                # while True:
                task_id, dem_index = self.get_random_demonstration(task, train=train)
                # print np.shape(joints), np.shape(images['camera-0'])
                joints = np.load(os.path.join(self.dest_dataset_path, task_id, str(dem_index) + '-joints.npy'))
                attention = np.load(os.path.join(self.out_filepath_attention, task_id, dem_index, 'camera-1.npy'))
                # if self.shuffle_att:
                #     np.random.shuffle(attention)
                # if len(joints) > self.step_size * self.sequence_size:

                # appending the beggining of the sequence with (joint_history) * self.step_size copies of the first joint angle
                first_joint = np.expand_dims(joints[0], axis=0)
                first_joint = np.repeat(first_joint, joint_history * self.step_size, axis=0)
                joints = np.concatenate((first_joint, joints), axis=0)
                # appending the end of the sequence with self.sequence_size * self.step_size copies of the last joint angle
                joints_len = len(joints)

                last_joint = np.expand_dims(joints[-1], axis=0)
                last_joint = np.repeat(last_joint, self.sequence_size * self.step_size, axis=0)
                joints = np.concatenate((joints, last_joint), axis=0)

                last_attention = np.expand_dims(attention[-1], axis=0)
                last_attention = np.repeat(last_attention, self.sequence_size * self.step_size, axis=0)
                attention = np.concatenate((attention, last_attention), axis=0)

                if joints_len > ((joint_history + self.sequence_size) * self.step_size):
                    robot_rand_index = np.random.randint(joints_len - ((joint_history + self.sequence_size) * self.step_size), size=1)[0]
                else:
                    robot_rand_index = 0   
                coin_toss = random.uniform(0, 1)
                if coin_toss < from_start_prob:
                    from_start = True
                if from_start:
                    robot_rand_index = 0

                # if robot_rand_index + self.sequence_size * self.step_size > joints_len:
                #     last_joint = np.expand_dims(joints[-1], axis=0)
                #     last_joint = np.repeat(last_joint, (robot_rand_index + self.sequence_size * self.step_size) - joints_len, axis=0)
                #     joints = np.concatenate((joints, last_joint), axis=0)
                # break
                robot_images_start_index = robot_rand_index + joint_history * self.step_size
                
                stage = float(robot_images_start_index) / joints_len
                stage = int(stage * 10)
                # for a in range(robot_images_start_index, robot_images_start_index + self.step_size * self.sequence_size, self.step_size):
                #     c = float(a) / joints_len
                #     c = int(c * 10)
                #     completeness[i, (a - robot_images_start_index)/self.step_size, c] = 1

                original_robot_images[i], robot_path, camera_used = self.read_robot_npy_batch(task_id, dem_index, camera,
                                                         joints[robot_images_start_index: robot_images_start_index + self.step_size * self.sequence_size, 0],  use_vgg=use_vgg)
                # robot_images_secondary[i], _, _ = self.read_robot_npy_batch(task_id, dem_index, camera_used,
                #                                          joints[robot_images_start_index: robot_images_start_index + self.step_size * self.sequence_size, 0], not_this_camera=True, use_vgg=use_vgg)
                robot_attentions[i] = attention[robot_rand_index: robot_rand_index + self.step_size * self.sequence_size: self.step_size]
                robot_paths.append(robot_path)
                robot_images[i] = np.copy(original_robot_images[i])
                if camera_used == 'camera-1':
                    robot_images_temp, toRet_robot_attentions, toRet_robot_cropped_attentions_temp = self.morph_attention_and_image(robot_images[i][np.newaxis], robot_attentions[i])
                    robot_images[i] = robot_images_temp[0]
                    toRet_robot_cropped_attentions[i] = toRet_robot_cropped_attentions_temp[0]

                if sth_sth:
                    sth_sth_images[i], sth_sth_path = self.read_sth_sth_npy_batch(task_id)
                    sth_sth_paths.append(sth_sth_path)

                if human:
                    human_images[i], human_path = self.read_human_npy_batch(task_id, dem_index, camera)
                    human_paths.append(human_path)

                # history_index = robot_rand_index - joint_history * self.step_size
                # if history_index >= 0:
                #     batch_joints[i] = joints[history_index: robot_rand_index + self.step_size * self.sequence_size: self.step_size]
                # else:
                #     half = joints[0: robot_rand_index + self.step_size * self.sequence_size: self.step_size]
                #     # print self.sequence_size + joint_history - len(half), -history_index/self.step_size
                #     #[1.0, 0.7670000195503235, 0.7925000190734863, 0.6265000104904175, 0.4659999907016754, 0.6579999923706055, 0.0]
                #     to_rep = np.asarray([[0.0 ,0.0 ,0.0 ,1.0, 0.7670000195503235, 0.7925000190734863, 0.6265000104904175, 0.4659999907016754,
                #                  0.6579999923706055, 0.0]])
                #     ohalf = np.repeat(to_rep, self.sequence_size + joint_history - len(half), axis=0)
                #     # ohalf = np.zeros((self.sequence_size + joint_history - len(half), self.csv_col_num))
                #     batch_joints[i] = np.concatenate((ohalf, half), axis=0)

                batch_joints[i] = joints[robot_rand_index: robot_rand_index + (self.sequence_size + joint_history) * self.step_size: self.step_size]

                if int(task_id) in self.annotated_tasks:
                    label, which_object, which_describtor, wrong_which_object, wrong_which_describtor = self.get_attention_label(task_id, dem_index)
                    object_involved[i] = which_object
                    object_involved_one_hot[i, which_object - 1] = 1
                    wrong_object_involved_one_hot[i, wrong_which_object - 1] = 1
                    describtors_involved[i] = which_describtor
                    describtors_involved_one_hot[i, which_describtor - 1] = 1
                    wrong_describtors_involved_one_hot[i, wrong_which_describtor - 1] = 1
                    attention_sentence[i] = label
            
            batch_one_hot = self.get_task_one_hot_vector(batch_joints)
            # original_robot_images = np.copy(robot_images)
            # if camera_used == 'camera-1':
            #     robot_images, toRet_robot_attentions = self.morph_attention_and_image(robot_images, robot_attentions)
            to_ret_human_images = human_images
            to_ret_robot_images = robot_images
            to_ret_original_robot_images = original_robot_images
            to_ret_robot_images_secondary = robot_images_secondary
            to_ret_sth_sth_images = sth_sth_images
            if not channel_first:
                to_ret_robot_images = np.swapaxes(robot_images, 2, -1)
                to_ret_robot_images = np.swapaxes(to_ret_robot_images, 2, 3)

                to_ret_original_robot_images = np.swapaxes(original_robot_images, 2, -1)
                to_ret_original_robot_images = np.swapaxes(to_ret_original_robot_images, 2, 3)

                to_ret_robot_images_secondary = np.swapaxes(robot_images_secondary, 2, -1)
                to_ret_robot_images_secondary = np.swapaxes(to_ret_robot_images_secondary, 2, 3) 

                to_ret_human_images = np.swapaxes(human_images, 2, -1)
                to_ret_human_images = np.swapaxes(to_ret_human_images, 2, 3)

                to_ret_sth_sth_images = np.swapaxes(sth_sth_images, 2, -1)
                to_ret_sth_sth_images = np.swapaxes(to_ret_sth_sth_images, 2, 3)

            if train:
                noisy_joints = self.apply_noise(task_id, batch_joints[:, :, 3:])
            else:
                noisy_joints = batch_joints[:, :, 3:]

            yield to_ret_robot_images, to_ret_original_robot_images, toRet_robot_cropped_attentions, to_ret_robot_images_secondary, toRet_robot_attentions, robot_paths, \
                to_ret_human_images, human_paths, \
                to_ret_sth_sth_images, sth_sth_paths, \
                batch_joints[:, :, 3:], noisy_joints, batch_one_hot, attention_sentence, \
                object_involved, object_involved_one_hot, describtors_involved, describtors_involved_one_hot, wrong_object_involved_one_hot, wrong_describtors_involved_one_hot

    def read_robot_npy_batch(self, task_id, dem_index, camera, timestamps, not_this_camera=False, use_vgg=False):
        if camera == 'random':
            cam_num = np.random.randint(len(self.cameras), size=1)[0]
            camera = 'camera-' + str(cam_num)

        if not_this_camera:
            camera2 = camera
            while camera == camera2:
                cam_num = np.random.randint(len(self.cameras), size=1)[0]
                camera2 = 'camera-' + str(cam_num)
            camera = camera2

        images = np.empty((self.sequence_size, self.num_channels, self.image_size, self.image_size))
        npys_path = os.path.join(self.dest_dataset_path, task_id, dem_index, 'robot', camera)
        # print [str(x) for x in timestamps]
        # print len(timestamps)
        for i in range(0, len(timestamps), self.step_size):
            # print ts
            if self.read_jpgs:
                path = os.path.join(npys_path, str(int(timestamps[i])) + '.jpg')
                image = self.read_image(path)
                images[int(i / self.step_size)] = self.pre_process_image(image, vgg=use_vgg)
            else:
                path = os.path.join(npys_path, str(int(timestamps[i])) + '.npy')
                images[int(i / self.step_size)] = np.load(path)
        paths = np.fromstring(path[::-1].zfill(200)[::-1], dtype=np.uint8)

        return images, paths, camera

    def read_sth_sth_npy_batch(self, task_id):
        images = np.empty((self.sequence_size, self.num_channels, self.image_size, self.image_size))
        task_path = os.path.join(self.dest_sth_sth_path, task_id)
        
        dems = [name for name in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, name))]
        dem_index = np.random.randint(len(dems), size=1)[0]
        images_list = os.listdir(os.path.join(task_path, dems[dem_index]))
        rand_index = np.random.randint(len(images_list) - self.sequence_size, size=1)[0]
        count = 0
        while count < self.sequence_size:
            if self.read_jpgs:
                path = os.path.join(os.path.join(task_path, dems[dem_index]), str(int(rand_index + count + 1)).zfill(5) + '.jpg')
                image = self.read_image(path)
                images[count] = self.pre_process_image(image)
            else:
                path = os.path.join(os.path.join(task_path, dems[dem_index]), str(int(rand_index + count + 1)) + '.npy')
                images[count] = np.load(path)
            
            count += 1
        paths = np.fromstring(path[::-1].zfill(200)[::-1], dtype=np.uint8)

        return images, paths

    def read_human_npy_batch(self, task_id, dem_index, camera):
        if camera == 'random':
            cam_num = np.random.randint(len(self.cameras), size=1)[0]
            camera = 'camera-' + str(cam_num)

        images = np.empty((self.sequence_size, self.num_channels, self.image_size, self.image_size))
        npys_path = os.path.join(self.dest_dataset_path, task_id, dem_index, 'human', camera)
        filename_list = sorted(os.listdir(npys_path))
        rand_index = np.random.randint(len(filename_list) - self.sequence_size, size=1)[0]
        count = 0
        index = 0

        while count < self.sequence_size:
            to_read_image = os.path.join(npys_path, filename_list[rand_index + count])
            if os.path.isfile(to_read_image):
                image = self.read_image(to_read_image)
                images[index] = self.pre_process_image(image)
                # images[index] = np.load(to_read_image)
                index += 1
            count += 1

        paths = np.fromstring(to_read_image[::-1].zfill(200)[::-1], dtype=np.uint8)

        return images, paths

    def process_sth_sth(self):
        for dir_name in os.listdir(self.source_sth_sth_path):
            if os.path.isdir(os.path.join(self.source_sth_sth_path, dir_name)):
                print('Found directory in sth sth: %s' % dir_name)
                self.process_sth_sth_tasks(dir_name)

    def process_dataset(self):
        for dir_name in os.listdir(self.source_dataset_path):
            if os.path.isdir(os.path.join(self.source_dataset_path, dir_name)) and int(dir_name) in self.reverse_annotations.keys():
                print('Found directory: %s' % dir_name)
                self.process_tasks(dir_name)

    def process_sth_sth_tasks(self, dir_name):
        source = os.path.join(self.source_sth_sth_path, dir_name)
        dest = os.path.join(self.dest_sth_sth_path, dir_name)
        create_dir(os.path.join(self.dest_sth_sth_path, dir_name))

        for subdir_name in os.listdir(source):
            if os.path.isdir(os.path.join(source, subdir_name)):
                create_dir(os.path.join(dest, subdir_name))
                self.process_sth_sth_demonstration(os.path.join(source, subdir_name), os.path.join(dest, subdir_name))

    def process_tasks(self, dir_name):
        source = os.path.join(self.source_dataset_path, dir_name)
        dest = os.path.join(self.dest_dataset_path, dir_name)
        create_dir(os.path.join(self.dest_dataset_path, dir_name))

        for subdir_name in os.listdir(source):
            if subdir_name in self.reverse_annotations[int(dir_name)].keys():
                create_dir(os.path.join(dest, subdir_name))
                self.process_demonstration(os.path.join(source, subdir_name), os.path.join(dest, subdir_name))

    def process_sth_sth_demonstration(self, dir_path, dest_path):
        # Reading the actual images

        for image_name in os.listdir(dir_path):
            if not os.path.isfile(os.path.join(dest_path, str(int(image_name[:-4])) + '.npy')):
                image = self.read_image(os.path.join(dir_path, image_name))
                image = self.pre_process_image(image)
                np.save(os.path.join(dest_path, str(int(image_name[:-4])) + '.npy'), image)

    def process_demonstration(self, dir_path, dest_path):
        create_dir(os.path.join(dest_path, 'robot'))
        create_dir(os.path.join(dest_path, 'human'))

        dem_folder = dir_path
        dem_csv = dir_path + '.txt'
        # print dem_csv
        joint_pos = pd.read_csv(dem_csv, header=2, sep=',', index_col=False)

        timestamps_robot = list(joint_pos['timestamp'])
        for camera in self.cameras:
            create_dir(os.path.join(dest_path, 'robot', camera))
            create_dir(os.path.join(dest_path, 'human', camera))
        # Finding timesstamps that are present in all cameras
        for camera in self.cameras:
            _, valid_ts = self.images_to_npy(dem_folder, dest_path, camera, timestamps=timestamps_robot, verify=True)
            timestamps_robot = valid_ts

        # Reading the actual images
        for camera in self.cameras:
            if not os.path.isfile(dest_path + '-joints.npy'):
                self.images_to_npy(dem_folder, dest_path, camera, timestamps=timestamps_robot, verify=False)
                self.images_to_npy(dem_folder, dest_path, camera, isRobot='human', timestamps=None, verify=False)
                # print 'folder ' + dem_folder + ' ' + camera + ' has ' + str(missing) + ' images/timestamps ' + \
                #       str(len(valid_ts)) + ' ' + str(len(images))

                # np.savez_compressed(dest_path + '-' + camera, images=images)
                # else:
                # print dest_path + '-' + camera + ' already compressed'

        joint_com_list = joint_pos.loc[joint_pos['timestamp'].isin(valid_ts)].values.tolist()
        np.save(dest_path + '-joints', joint_com_list)

    def images_to_npy(self, path, dest_path, camera, timestamps=None, isRobot='robot', verify=False):
        images = []
        valid_ts = []
        if timestamps is None:
            images_path = os.path.join(path, isRobot, camera)
            all_files = os.listdir(images_path)
            for file_in_use in all_files:
                to_read = os.path.join(images_path, file_in_use)
                if os.path.isfile(to_read):
                    if not verify:
                        image = self.read_image(to_read)
                        image = self.pre_process_image(image)
                        np.save(os.path.join(dest_path, isRobot, camera, file_in_use[:-4] + '.npy'), image)
                    valid_ts.append(file_in_use[:-4])
        else:
            for ts in timestamps:
                to_read = os.path.join(path, isRobot, camera, str(ts) + '.jpg')
                # print to_read
                if os.path.isfile(to_read):
                    if not verify:
                        image = self.read_image(to_read)
                        image = self.pre_process_image(image)
                        np.save(os.path.join(dest_path, isRobot, camera, str(ts) + '.npy'), image)
                    valid_ts.append(ts)

        return images, valid_ts

    def read_image(self, path):
        image = Image.open(path)
        camera_id = 1
        if "camera-0" in path: 
            camera_id = 0
        elif "camera-2" in path:
            camera_id = 2

        image = self.transformer.apply_homography(image, camera_id)
        # image.show()
        # image = []
        return image

    def pre_process_image(self, image, vgg=False):
        if vgg:
            return self.prepare_vgg(image)

        if self.num_channels == 1:
            image = image.convert('L')

        if self.num_channels == 3:
            image = image.convert('RGB')
            image = np.array(image)
            image = image.transpose((2, 0, 1))
            image = image[:, :, ::-1].copy()

        image = np.asarray(image, dtype=np.float32)
        image = image / 127.5 - 1
        # image = image / 255
        return image

    def prepare_vgg(self, image, size=(224, 224)):
        """Converts the given image to the np array for VGG models.
        Note that you have to call this method before ``__call__``
        because the pre-trained vgg model requires to resize the given image,
        covert the RGB to the BGR, subtract the mean,
        and permute the dimensions before calling.
        Args:
            image (PIL.Image or np.ndarray): Input image.
                If an input is ``np.ndarray``, its shape must be
                ``(height, width)``, ``(height, width, channels)``,
                or ``(channels, height, width)``, and
                the order of the channels must be RGB.
            size (pair of ints): Size of converted images.
                If ``None``, the given image is not resized.
        Returns:
            np.ndarray: The converted output array.
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                if image.shape[0] == 1:
                    image = image[0, :, :]
                elif image.shape[0] == 3:
                    image = image.transpose((1, 2, 0))
            image = Image.fromarray(image.astype(np.uint8))
        image = image.convert('RGB')
        if size:
            image = image.resize(size)
        image = np.asarray(image, dtype=np.float32)
        image = image[:, :, ::-1]
        image -= np.array(
            [103.939, 116.779, 123.68], dtype=np.float32)
        image = image.transpose((2, 0, 1))
        return image

def show_image(images):
    # for i in range(images.shape[0]):
        # moving axis to use plt: i.e [4,100,100] to [100,100,4]
    img = images
    img = img.transpose(1, 2, 0)
    img = (img +1) *127.5
    img = img.astype(np.uint8)
    print img.dtype, np.max(img), np.min(img), np.shape(img)
    img = Image.fromarray(img, "RGB")
    img.show()

if __name__ == '__main__':
    DC = DatasetController(batch_size=6, sequence_input=1, sequence_output=0, read_jpgs=True)
    # g = DC.get_next_batch(task='5002', channel_first=False, human=False, sth_sth=True, joint_history=10)
    g1 = DC.get_next_batch(task=['5001', '5002'], channel_first=True, human=False, from_start_prob=0, joint_history=0, camera='camera-1')

    while True:
        start = time.time()
        # robot_images, robot_paths, human_images, human_paths, sth_sth_images, sth_sth_paths, joints, noisy_joints, batch_one_hot, attention_labels, attention_gt = next(g)
        to_ret_robot_images, to_ret_robot_images_orginals, to_ret_robot_images_cropped, robot_images_secondary, attention, robot_paths, to_ret_human_images, human_paths, to_ret_sth_sth_images, sth_sth_paths, batch_joints, noisy_joints, batch_one_hot, attention_sentence, objects, obj_one_hot, descriptions, desc_one_hot, wrong_obj_one_hot, wrong_desc_one_hot = next(g1)
        # show_image(to_ret_robot_images_orginals[1,0])
        # robot_images, robot_paths, human_images, human_paths, sth_sth_images, sth_sth_paths, joints = DC.get_next_batch(task='5001', sth_sth=True, human=True)
        # print(np.shape(robot_images), np.shape(human_images), np.shape(sth_sth_images), np.shape(joints))
        # print(np.shape(robot_paths), np.shape(human_paths), np.shape(sth_sth_paths))

        print(time.time() - start)
