from dataset.sonar import augment_pil
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
        self.teacher_mode = teacher_mode  # 添加教师模式参数

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
        
        # 生成标准弱增强和两种声纳弱增强
        img_w_standard = deepcopy(img)   # 标准弱增强（基本不变）
        img_w_sonar_a = deepcopy(img)    # 声纳A弱增强
        img_w_sonar_b = deepcopy(img)    # 声纳B弱增强
        img_s1, img_s2 = deepcopy(img), deepcopy(img)

        # Teacher A: acoustic shadow only
        img_w_sonar_a = augment_pil(img_w_sonar_a, 'sonar_a')
        
        # Teacher B: energy attenuation only
        img_w_sonar_b = augment_pil(img_w_sonar_b, 'sonar_b')

        # Paper Sec. 5.1: the same photometric strong views for every teacher.
        for strong in (1, 2):
            view = img_s1 if strong == 1 else img_s2
            if random.random() < 0.8:
                view = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(view)
            view = transforms.RandomGrayscale(p=0.2)(view)
            view = blur(view, p=0.5)
            if strong == 1:
                img_s1 = view
            else:
                img_s2 = view
        # CutMix is not specified by the paper; retain the loader interface only.
        cutmix_box1 = torch.zeros(self.size, self.size)
        cutmix_box2 = torch.zeros(self.size, self.size)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w_standard), normalize(img_w_sonar_a), normalize(img_w_sonar_b), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
