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


class DatasetController_multi_object:
    def __init__(self, batch_size, sequence_size, string_size=10, shuffle_att=False):
        self.tasks = [6001, 6002, 6003]
        self.task_description = {
            6001: 'push white plate from left to right',
            6002: 'push blue box from left to right',
            6003: 'push black-white qr-box from left to right',
        }
        self.sequence_size = sequence_size
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.cameras = ['camera-1']
        self.attention_size = 28
        self.train_percentage = 0.8
        self.shuffle_att = shuffle_att

        self.transformer = imageTransformer("empty")
        self.source_dataset_path = dataset_path
        self.source_multi_object_dataset_path = multi_object_dataset_path
        self.dest_multi_object_dataset_path = self.source_multi_object_dataset_path
        
        self.out_filepath_attention = './processed_inputs_multi_object_dense_28/'

        self.annotations = {}
        self.reverse_annotations = {}
        self.string_size = string_size
        self.annotated_tasks = [5001, 5002]
        self.bag_of_words = self.fill_bag_of_words(self.annotated_tasks)
        self.read_annotations()
        self.step_size = 3

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

        # self.objects_describtors = {"white" : 1,
        #                 "blue": 2,
        #                 "black-white": 3,
        #                 "black": 4,
        #                 "green": 5,
        #                 "red": 6,
        #                 "orange": 7}
        # self.objects = {"plate": 1,
        #                 "box": 2,
        #                 "qr-box": 3,
        #                 "bar": 4,
        #                 "remote": 5,
        #                 "spray": 6,
        #                 "headphone": 7,
        #                 "marker": 8,
        #                 "dental-floss": 9,
        #                 "flashlight": 10,
        #                 "wire": 11,
        #                 "usb": 12,
        #                 "screw-driver": 13,
        #                 "glasses": 14,
        #                 "ear-plug": 15}

        self.objects_describtors = {self.bag_of_words[x]:y for x,y in self.objects_describtors.iteritems()}
        self.objects = {self.bag_of_words[x]:y for x,y in self.objects.iteritems()}
        self.num_all_objects = len(self.objects.keys())
        self.num_all_objects_describtors = len(self.objects_describtors.keys())

        self.train_images = {}
        self.test_images = {}
        self.separate_train_test()

    def separate_train_test(self):
        for task_id in self.tasks:
            task_path = os.path.join(self.dest_multi_object_dataset_path, str(task_id))
            images_names = [name for name in os.listdir(os.path.join(task_path, "1", "camera-1"))]

            num_images = len(images_names)
            self.train_images[task_id] = images_names[:int(num_images * self.train_percentage)]
            self.test_images[task_id] = images_names[int(num_images * self.train_percentage):]

    def encode_annotation(self, sentence):
        words = sentence.split()
        encoded = np.zeros((self.string_size), dtype=int)
        for i, word in enumerate(words):
            encoded[i] = int(self.bag_of_words[word])
        
        return encoded

    def read_annotations(self):
        for task, value in self.task_description.iteritems():
            self.reverse_annotations[task] = {}
            words = value.split()
            key = ''
            for word in words:
                key += str(self.bag_of_words[word]) + ' '
            key += '0'
            self.annotations[key] = '1'
            self.reverse_annotations[task]['1'] = key

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

    def get_random_demonstration(self, train=True):
        # print len(self.tasks.keys()), self.tasks.keys()[rand_task]

        rand_task = np.random.randint(len(self.tasks), size=1)[0]
        task_id = self.tasks[rand_task]
        if train:
            folders =  self.train_images[task_id]
        else:
            folders =  self.test_images[task_id]

        num_dems = len(folders)
        rand_dem = np.random.randint(num_dems, size=1)[0]

        return task_id, folders[rand_dem]

    def get_attention_label(self, task):
        correct_sentence = self.reverse_annotations[int(task)]['1']
        
        wrong_sentences = [item for item in self.annotations.keys() if item not in [correct_sentence]]
        random_wrong_key = np.random.randint(len(wrong_sentences), size=1)[0]

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
        toRet_cropped_att = np.empty((images.shape[0], 1, 3, self.image_size, self.image_size))
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                att_orig = np.reshape(attentions[j], (28, 28))
                att = F.max_pooling_2d(Variable(att_orig[np.newaxis, np.newaxis, :]), 7).data
                att[att < 0.95] = 0
                if self.shuffle_att:
                    att[:] = 0
                    rand_w, rand_h= np.random.randint(att.shape[-1], size=2)
                    att[0,0,rand_w, rand_h] = 1
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
        
        return images, toRet_cropped_att

    def get_next_batch(self, train=True, camera='camera-1', use_vgg=False):
        original_robot_images = np.empty(
            (self.batch_size * self.sequence_size, 1, self.num_channels, self.image_size, self.image_size))
        robot_images = np.empty(
            (self.batch_size * self.sequence_size, 1, self.num_channels, self.image_size, self.image_size))
        robot_attentions = np.empty(
            (self.batch_size * self.sequence_size, 1, self.attention_size, self.attention_size))
        toRet_robot_cropped_attentions = np.empty(
            (self.batch_size * self.sequence_size, 1, 3, self.image_size, self.image_size))

        attention_sentence = np.zeros((self.batch_size * self.sequence_size, self.string_size))
        object_involved = np.zeros((self.batch_size * self.sequence_size))
        describtors_involved = np.zeros((self.batch_size * self.sequence_size))

        object_involved_one_hot = np.zeros((self.batch_size * self.sequence_size, self.num_all_objects))
        describtors_involved_one_hot = np.zeros((self.batch_size * self.sequence_size, self.num_all_objects_describtors))
        wrong_object_involved_one_hot = np.zeros((self.batch_size * self.sequence_size, self.num_all_objects))
        wrong_describtors_involved_one_hot = np.zeros((self.batch_size * self.sequence_size, self.num_all_objects_describtors))

        while True:
            for i in range(self.batch_size * self.sequence_size):
                task_id, dem_index = self.get_random_demonstration(train=train)
                # print np.shape(joints), np.shape(images['camera-0'])
                attention_dic = np.load(os.path.join(self.out_filepath_attention, str(task_id), '1', 'camera-1.npy')).item()
                
                atts = attention_dic[dem_index]
                robot_attentions[i, 0] = atts

                image = self.read_image(os.path.join(self.source_multi_object_dataset_path, str(task_id), '1', 'camera-1', dem_index))
                original_robot_images[i, 0] = self.pre_process_image(image, vgg=use_vgg)

                robot_images[i] = np.copy(original_robot_images[i])
                robot_images_temp, robot_cropped_temp = self.morph_attention_and_image(robot_images[i][np.newaxis], robot_attentions[i])
                robot_images[i] = robot_images_temp[0]
                toRet_robot_cropped_attentions[i] = robot_cropped_temp[0]

                label, which_object, which_describtor, wrong_which_object, wrong_which_describtor = self.get_attention_label(task_id)
                object_involved[i] = which_object
                object_involved_one_hot[i, which_object - 1] = 1
                wrong_object_involved_one_hot[i, wrong_which_object - 1] = 1
                describtors_involved[i] = which_describtor
                describtors_involved_one_hot[i, which_describtor - 1] = 1
                wrong_describtors_involved_one_hot[i, wrong_which_describtor - 1] = 1
                attention_sentence[i] = label
            
            to_ret_robot_images = robot_images
            to_ret_original_robot_images = original_robot_images

            to_ret_robot_images_reshaped = np.reshape(to_ret_robot_images, (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            to_ret_original_robot_images_reshaped = np.reshape(to_ret_original_robot_images, (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            toRet_robot_cropped_attentions_reshaped = np.reshape(toRet_robot_cropped_attentions, (self.batch_size, self.sequence_size, self.num_channels, self.image_size, self.image_size))
            attention_sentence_reshaped = np.reshape(attention_sentence, (self.batch_size, self.sequence_size, self.string_size))
            object_involved_reshaped = np.reshape(object_involved, (self.batch_size, self.sequence_size))
            object_involved_one_hot_reshaped = np.reshape(object_involved_one_hot, (self.batch_size, self.sequence_size, self.num_all_objects))
            describtors_involved_reshaped = np.reshape(describtors_involved, (self.batch_size, self.sequence_size))
            describtors_involved_one_hot_reshaped = np.reshape(describtors_involved_one_hot, (self.batch_size, self.sequence_size, self.num_all_objects_describtors))
            wrong_object_involved_one_hot_reshaped = np.reshape(wrong_object_involved_one_hot, (self.batch_size, self.sequence_size, self.num_all_objects))
            wrong_describtors_involved_one_hot_reshaped = np.reshape(wrong_describtors_involved_one_hot, (self.batch_size, self.sequence_size, self.num_all_objects_describtors))

            yield to_ret_robot_images_reshaped, to_ret_original_robot_images_reshaped, toRet_robot_cropped_attentions_reshaped, \
                attention_sentence_reshaped, object_involved_reshaped, object_involved_one_hot_reshaped, describtors_involved_reshaped, \
                describtors_involved_one_hot_reshaped, wrong_object_involved_one_hot_reshaped, wrong_describtors_involved_one_hot_reshaped

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
    DC = DatasetController_multi_object(batch_size=6, sequence_size=10)
    # g = DC.get_next_batch(task='5002', channel_first=False, human=False, sth_sth=True, joint_history=10)
    g1 = DC.get_next_batch(train=True, camera='camera-1')

    while True:
        start = time.time()
        # robot_images, robot_paths, human_images, human_paths, sth_sth_images, sth_sth_paths, joints, noisy_joints, batch_one_hot, attention_labels, attention_gt = next(g)
        to_ret_robot_images, to_ret_original_robot_images, to_ret_cropped_robot_images, attention_sentence, object_involved, object_involved_one_hot, describtors_involved, describtors_involved_one_hot, wrong_object_involved_one_hot, wrong_describtors_involved_one_hot = next(g1)

        # show_image(to_ret_robot_images[0,0])
        # show_image(to_ret_original_robot_images[0,0])
        # robot_images, robot_paths, human_images, human_paths, sth_sth_images, sth_sth_paths, joints = DC.get_next_batch(task='5001', sth_sth=True, human=True)
        # print(np.shape(robot_images), np.shape(human_images), np.shape(sth_sth_images), np.shape(joints))
        # print(np.shape(robot_paths), np.shape(human_paths), np.shape(sth_sth_paths))

        print(time.time() - start)
