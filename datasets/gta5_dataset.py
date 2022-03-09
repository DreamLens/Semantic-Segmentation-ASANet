import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
from torch.utils import data
import cv2
from PIL import Image


class GTA5DataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, img_size=(321, 321), norm=False, random_mirror=False, random_crop=False, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.img_size = img_size
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.norm = norm
        self.random_mirror = random_mirror
        self.random_crop = random_crop
        if not max_iters == None:
            self.img_ids = self.img_ids * \
                int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "images/%s" % name)
            label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "lbl": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        name = datafiles["name"]
        image = Image.open(datafiles["img"]).convert("RGB")
        label = Image.open(datafiles["lbl"])
        if self.random_crop:
            img_w, img_h = image.size
            crop_w, crop_h = self.img_size
            if img_h < crop_h or img_w < crop_w:
                image = image.resize(self.img_size, Image.BICUBIC)
                label = label.resize(self.img_size, Image.NEAREST)
            else:
                h_off = random.randint(0, img_h - crop_h)
                w_off = random.randint(0, img_w - crop_w)
             