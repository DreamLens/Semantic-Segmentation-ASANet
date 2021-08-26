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
        self.img_ids