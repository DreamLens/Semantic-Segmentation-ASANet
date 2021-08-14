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
    def __init__(self, root, list_path, max_iters=None, img_size=(321, 321), norm=False, random