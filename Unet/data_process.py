import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import optim, nn
from torchvision.transforms import v2 as tf
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchmetrics.segmentation import DiceScore
import os
from PIL import Image
from tqdm.auto import tqdm

transforms = {
    'train' : tf.Compose(
        [
            tf.ToImage(),
            tf.Resize((256, 256)),
            tf.RandomHorizontalFlip(),
            tf.RandomVerticalFlip()
        ]
    ),
    'val' : tf.Compose(
        [
            tf.ToImage(),
            tf.Resize((256, 256)),
            tf.RandomHorizontalFlip(),
            tf.RandomVerticalFlip()
        ]
    ),
    'test' : tf.Compose(
        [
            tf.ToImage(),
            tf.Resize((256, 256))
        ]
    )
}

class GetDataset(Dataset):

    def __init__(
            self,
            base_dir,
            data_dir_list,
            transforms = None
    ):
        self.base_dir = base_dir
        self.data_dir_list = data_dir_list
        self.transforms = transforms
        self.img_dir_list = [
            os.path.join(self.base_dir, data_dir, 'image')
            for data_dir in self.data_dir_list
        ]
        self.mask_dir_list = [
            os.path.join(self.base_dir, data_dir, 'mask')
            for data_dir in self.data_dir_list
        ]
        self.img_paths = self._get_img_paths()
        self.mask_paths = self._get_mask_paths()
        self._convert_dtype = tf.ToDtype(torch.float32, scale=True)
        self._normalize = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _get_img_paths(self):
        img_paths = []
        for img_dir in self.img_dir_list:
            for img in os.scandir(img_dir):
                if img.is_file():
                    img_paths.append(
                        os.path.join(img_dir, img.name)
                    )
        return sorted(img_paths)

    def _get_mask_paths(self):
        mask_paths = []
        for mask_dir in self.mask_dir_list:
            for mask in os.scandir(mask_dir):
                if mask.is_file():
                    mask_paths.append(
                        os.path.join(mask_dir, mask.name)
                    )
        return sorted(mask_paths)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = np.array(Image.open(img_path))
        # Only the first channel contains the segmentation annotations, rest are all zeros
        mask = np.array(Image.open(mask_path))[:, :]
        if self.transforms:
            img, mask = self.transforms(img, mask)
        img = self._convert_dtype(img)
        mask = mask.squeeze(0)
        return img, mask
    
class test_GetDataset(Dataset):

    def __init__(
            self,
            base_dir,
            data_dir_list,
            transforms = None
    ):
        self.base_dir = base_dir
        self.data_dir_list = data_dir_list
        self.transforms = transforms
        self.img_dir_list = [
            os.path.join(self.base_dir, data_dir)
            for data_dir in self.data_dir_list
        ]
        self.img_paths = self._get_img_paths()
        self._convert_dtype = tf.ToDtype(torch.float32, scale=True)
        self._normalize = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _get_img_paths(self):
        img_paths = []
        for img_dir in self.img_dir_list:
            for img in os.scandir(img_dir):
                if img.is_file():
                    img_paths.append(
                        os.path.join(img_dir, img.name)
                    )
        return sorted(img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = np.array(Image.open(img_path))
        # Only the first channel contains the segmentation annotations, rest are all zeros
        if self.transforms:
            img = self.transforms(img)
        img = self._convert_dtype(img)
        return img