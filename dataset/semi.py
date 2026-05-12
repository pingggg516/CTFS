from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None, teacher_mode='supervised'):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.teacher_mode = teacher_mode

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open(os.path.join('splits', name, 'val.txt'), 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        if self.mode == 'train_u':
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)


        img_w_standard = deepcopy(img)
        img_w_sonar = deepcopy(img)
        img_s1, img_s2 = deepcopy(img), deepcopy(img)


        img_w_sonar = sonar_weak_augmentation(img_w_sonar)


        if self.teacher_mode == 'supervised':


            if random.random() < 0.8:
                img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
            img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
            img_s1 = blur(img_s1, p=0.5)
            cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)


            if random.random() < 0.8:
                img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
            img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
            img_s2 = blur(img_s2, p=0.5)
            cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        elif self.teacher_mode == 'general':


            if random.random() < 0.8:
                img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
            img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
            img_s1 = blur(img_s1, p=0.5)
            cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)


            if random.random() < 0.8:
                img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
            img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
            img_s2 = blur(img_s2, p=0.5)
            cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        elif self.teacher_mode == 'sonar':

            img_s1 = sonar_strong_augmentation(img_s1)
            cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

            img_s2 = sonar_strong_augmentation(img_s2)
            cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)
        else:
            raise ValueError(f"Unknown teacher_mode: {self.teacher_mode}")

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w_standard), normalize(img_w_sonar), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
