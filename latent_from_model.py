import numpy as np
import time
import sys
import os
import copy
import chainer.functions as F
import signal
import pandas as pd
from PIL import Image
import threading

from gpu import GPU
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
import autoencoders.tower
from local_config import config
from image_transformer import imageTransformer

from DatasetController_morph import DatasetController

def signal_handler(signal, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

locals().update(config)


def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


class EncodeDataset:
    def __init__(self):
        self.model_path = "model/"
        self.out_filepath = './processed_inputs/'

        create_dir(self.out_filepath)

        self.image_size = image_size
        self.latent_size = latent_size
        self.batch_size = 100
        self.sequence_size = 1
        self.dataset_path = dataset_path
        self.source_dataset_path = dataset_path
        self.num_channels = num_channels
        self.cameras_to_process = ['camera-1']
        self.cameras = self.cameras_to_process
        self.tasks_to_process = ['5001', '5002']

        self.transformer = imageTransformer("empty")
        self.use_all_cameras_stacked_on_channel = False
        if self.use_all_cameras_stacked_on_channel:
            self.num_channels = self.num_channels * 3

        self.batch_gpu_threads = [None] * GPU.num_gpus

        self.DC = DatasetController(batch_size=32, sequence_input=1, sequence_output=0, read_jpgs=True)

        self.num_descs = self.DC.num_all_objects_descriptions
        self.num_objects = self.DC.num_all_objects

        self.enc_model = autoencoders.tower.Encoder_text_tower(density=8, size=image_size, latent_size=latent_size, channel=self.num_channels, num_objects=self.num_objects, num_descriptions=self.num_descs)

        self.load_model()
        self.to_gpu()

        self.process_dataset()

    def encode(self, image, object_involved_one_hot, descriptions_involved_one_hot):
        xp = cuda.cupy
        cuda.get_device(GPU.main_gpu).use()

        gpu_batch_size = 1
        
        x_in = cuda.to_gpu(image, GPU.main_gpu)

        object_involved_one_hot = np.repeat(object_involved_one_hot, x_in.shape[0], axis=0)
        descriptions_involved_one_hot = np.repeat(descriptions_involved_one_hot, x_in.shape[0], axis=0)
        z0, mean, var = self.enc_model(Variable(x_in), Variable(cuda.to_gpu(object_involved_one_hot, GPU.main_gpu)), Variable(cuda.to_gpu(descriptions_involved_one_hot, GPU.main_gpu)) ,train=False)
        # print z0.shape
        return F.squeeze(z0)

    def process_dataset(self):
        for dir_name in os.listdir(self.source_dataset_path):
            if os.path.isdir(os.path.join(self.source_dataset_path, dir_name)):
                print('Found directory: %s' % dir_name)
                if dir_name in self.tasks_to_process:
                    if int(dir_name) in self.DC.reverse_annotations.keys():
                        self.process_tasks(dir_name)

    def process_tasks(self, dir_name):
        source = os.path.join(self.source_dataset_path, dir_name)
        dest = os.path.join(self.out_filepath, dir_name)
        create_dir(dest)

        for subdir_name in os.listdir(source):
            if subdir_name in self.DC.reverse_annotations[int(dir_name)].keys():
                dest_subdir_path = os.path.join(dest, subdir_name)
                create_dir(dest_subdir_path)
                _, which_obj, which_desc, _, _ = self.DC.get_attention_label(dir_name, subdir_name)

                object_involved_one_hot = np.zeros((1, self.num_objects), dtype=np.float32)
                descriptions_involved_one_hot = np.zeros((1, self.num_descs), dtype=np.float32)
                object_involved_one_hot[0, which_obj - 1] = 1
                descriptions_involved_one_hot[0, which_desc - 1] = 1

                self.process_demonstration(os.path.join(source, subdir_name), dest_subdir_path, object_involved_one_hot, descriptions_involved_one_hot)

    def process_demonstration(self, dir_path, dest_dir_path, object_involved_one_hot, descriptions_involved_one_hot):
        dem_folder = dir_path
        dem_csv = dir_path + '.txt'
        # print dem_csv
        joint_pos = pd.read_csv(dem_csv, header=2, sep=',', index_col=False)

        timestamps_robot = list(joint_pos['timestamp'])
        #Finding timesstamps that are present in all cameras
        for camera in self.cameras:
            _, valid_ts = self.images_to_latent(dem_folder, camera, object_involved_one_hot, descriptions_involved_one_hot, timestamps=timestamps_robot, verify=True)
            timestamps_robot = valid_ts
        # _, valid_ts= self.all_camera_images_to_latent(dem_folder, timestamps=timestamps_robot, verify=True)
        # timestamps_robot = valid_ts
        # Reading the actual images

        for camera in self.cameras_to_process:
            if not os.path.exists(os.path.join(dest_dir_path, camera + '.npy')):
                latents, _ = self.images_to_latent(dem_folder, camera, object_involved_one_hot, descriptions_involved_one_hot ,timestamps=timestamps_robot, verify=False)
                np.save(os.path.join(dest_dir_path, camera + '.npy'), latents)

    def images_to_latent(self, path, camera, object_involved_one_hot, descriptions_involved_one_hot, timestamps=None, isRobot='robot', verify=False):
        valid_ts = []
        latents = np.empty((len(timestamps), self.latent_size))
        images = np.zeros((len(timestamps), self.num_channels, self.image_size, self.image_size), dtype=np.float32)
        for i, ts in enumerate(timestamps):
            to_read = os.path.join(path, isRobot, camera, str(ts) + '.jpg')
            # print to_read
            if os.path.isfile(to_read):
                if not verify:
                    image = self.read_image(to_read)
                    image = self.pre_process_image(image)
                    images[i] = np.asarray(image, dtype=np.float32)
                valid_ts.append(ts)

        if not verify:
            if images.shape[0] > self.batch_size:
                for j in range(0, images.shape[0] - (images.shape[0] % self.batch_size), self.batch_size):
                    ext = self.encode(images[j:j + self.batch_size], object_involved_one_hot, descriptions_involved_one_hot)
                    latents[j:j + self.batch_size] = cuda.to_cpu(ext.data)

                j += self.batch_size
                if j < images.shape[0]:
                    ext = self.encode(images[j:], object_involved_one_hot, descriptions_involved_one_hot)
                    latents[j:] = cuda.to_cpu(ext.data)
            else:
                ext = self.encode(images,  object_involved_one_hot, descriptions_involved_one_hot)
                latents = cuda.to_cpu(ext.data)

        # latents[i] = cuda.to_cpu(latent.data)
        return latents, valid_ts

    def load_model(self):
        try:
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model_path)
            print os.path.dirname(__file__)
            serializers.load_hdf5(file_path + 'enc.model', self.enc_model)
            # self.enc_models = [self.enc_model]
            #
            # for i in range(GPU.num_gpus - 1):
            #     self.enc_models.append(copy.deepcopy(self.enc_model))
        except Exception as inst:
            print inst
            print 'cannot load the encoder model from {}'.format(file_path)

    def to_gpu(self):
            self.enc_model.to_gpu(GPU.main_gpu)

    def read_image(self, path):
        image = Image.open(path)
        camera_id = 1
        if "camera-0" in path: 
            camera_id = 0
        elif "camera-2" in path:
            camera_id = 2

        image = self.transformer.apply_homography(image, camera_id)
        return image

    def pre_process_image(self, image):
        if self.num_channels == 1:
            image = image.convert('L')

        if self.num_channels >= 3:
            image = image.convert('RGB')
            image = np.array(image)
            image = image.transpose((2, 0, 1))
            image = image[:, :, ::-1].copy()

        image = np.asarray(image, dtype=np.float32)
        image = image / 127.5 - 1
        # image = image / 255
        return image

if __name__ == '__main__':
    ED = EncodeDataset()
