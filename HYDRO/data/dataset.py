from torchvision.transforms.functional import center_crop, gaussian_blur, resize
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import json
import os
import numpy as np
import random
import tqdm
from skimage import io
from PIL import Image


class DeIDDataset(Dataset):
    def __init__(self,
                 root_dir='',
                 transform=None,
                 iterations=5000000,
                 ):
        self.root_dir = root_dir
        self.transform = transform
        self.identities = os.listdir(root_dir)
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        target_identity = np.random.randint(len(self.identities))

        target_faces = os.listdir(os.path.join(self.root_dir, self.identities[target_identity]))

        target_face = np.random.randint(len(target_faces))

        try:
            # Image data
            target = np.array(Image.open(os.path.join(self.root_dir,
                                                      self.identities[target_identity],
                                                      target_faces[target_face])).convert('RGB'))
            if self.transform is not None:
                t = self.transform(image=target)
                target = t['image']

            return target

        except Exception as e:
            return self.__getitem__(np.random.randint(self.__len__()))


class GenerateEvalDataset(Dataset):
    def __init__(self,
                 root_dir='',
                 transform=None,
                 target_root_dir=None,
                 face_swap_mode=False):
        self.root_dir = root_dir.replace('/', '\\')
        self.transform = transform
        self.identities = os.listdir(root_dir)

        self.face_swap_mode = face_swap_mode

        self.sample_paths = []

        for root, dirs, files in tqdm.tqdm(os.walk(self.root_dir)):
            for name in files:
                if name[-4:] == '.png' or name[-4:] == '.jpg': self.sample_paths.append(os.path.join(root, name))

        self.target_root_dir = None
        if target_root_dir is not None:
            self.target_root_dir = target_root_dir.replace('/', '\\')

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        current_file_path = self.sample_paths[idx]
        file_path_splits = current_file_path.split('\\')
        current_id = file_path_splits[-2]
        current_file = file_path_splits[-1]

        # Get the face image
        target = Image.open(current_file_path)

        if self.transform is not None:
            target = self.transform(target)

        if self.target_root_dir is None:
            if not self.face_swap_mode:
                return target, current_file, current_id
            else:
                try:
                    source_id = self.identities[np.random.randint(len(self.identities))]

                    while source_id == current_id:
                        source_id = self.identities[np.random.randint(len(self.identities))]

                    source_files = os.listdir(os.path.join(self.root_dir, source_id))
                    source_file = source_files[np.random.randint(len(source_files))]

                    source = Image.open(os.path.join(self.root_dir, source_id, source_file))

                    if self.transform is not None:
                        source = self.transform(source)
                except Exception as e:
                    print(e)
                    print(source_id)
                    raise e

                return target, current_file, current_id, source, source_file, source_id
        else:
            unmanipulated_target = Image.open(current_file_path.replace(self.root_dir, self.target_root_dir))

            if self.transform is not None:
                unmanipulated_target = self.transform(unmanipulated_target)

            return target, unmanipulated_target, current_file, current_id

    def shuffle(self):
        random.shuffle(self.identities)


class EvalDataset(Dataset):
    def __init__(self,
                 target_root_dir='',
                 source_root_dir='',
                 transform=None,):
        self.target_root_dir = target_root_dir.replace('/', '\\')
        self.source_root_dir = source_root_dir.replace('/', '\\')
        self.transform = transform
        self.identities = os.listdir(target_root_dir)

        self.sample_paths = []

        for root, dirs, files in tqdm.tqdm(os.walk(self.target_root_dir)):
            for name in files:
                if name[-4:] == '.png' or name[-4:] == '.jpg': self.sample_paths.append(os.path.join(root, name))

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        current_file_path = self.sample_paths[idx]
        file_path_splits = current_file_path.split('\\')
        current_id = file_path_splits[-2]
        current_file = file_path_splits[-1]

        # Get the face image
        target = Image.open(current_file_path)
        source = Image.open(current_file_path.replace(self.target_root_dir, self.source_root_dir))

        if self.transform is not None:
            target = self.transform(target)
            source = self.transform(source)

        return target, source, current_file, current_id

    def shuffle(self):
        random.shuffle(self.identities)


class LFWProtocolDatasetGenuine(Dataset):
    def __init__(self,
                 root_dir_target='',
                 root_dir_genuine='',
                 transform=None,
                 protocol_file="pairsDevTest_genuine.txt",
                 imposter_mode=False):
        self.root_dir_target = root_dir_target.replace('/', '\\')
        self.root_dir_genuine = root_dir_genuine.replace('/', '\\')
        self.transform = transform
        self.identities = os.listdir(root_dir_target)
        self.imposter_mode = imposter_mode

        lfw_imposter_file = open(protocol_file, 'r')
        pairs = lfw_imposter_file.readlines()

        self.target_paths = []
        self.genuine_paths = []

        print("Mapping all file paths..")

        for pair in pairs:
            pair_split = pair.split('\t')
            if len(pair_split) == 3:
                try:
                    genuine_path = os.path.join(self.root_dir_genuine, pair_split[0])
                    genuine_index = int(pair_split[2]) - 1
                    genuine_image_list = os.listdir(genuine_path)

                    self.genuine_paths.append(os.path.join(genuine_path, genuine_image_list[genuine_index]))

                    target_path = os.path.join(self.root_dir_target, pair_split[0])
                    target_index = int(pair_split[1]) - 1
                    target_image_list = os.listdir(target_path)

                    self.target_paths.append(os.path.join(target_path, target_image_list[target_index]))

                except Exception as e:
                    print(e)
                    print(target_path)

    def __len__(self):
        return len(self.target_paths)

    def __getitem__(self, idx):
        target_file_path = self.target_paths[idx]
        genuine_file_path = self.genuine_paths[idx]

        target = Image.open(target_file_path)
        genuine = Image.open(genuine_file_path)

        if self.transform is not None:
            target = self.transform(target)
            genuine = self.transform(genuine)

        return genuine, target

    def shuffle(self):
        random.shuffle(self.identities)



