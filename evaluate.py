import os
import os.path as osp
from collections import OrderedDict
import sys
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pprint import pprint
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from networks.deeplab import Deeplab_Res101
from networks.deeplab_vgg import DeeplabVGG
from networks.fcn8s import VGG16_FCN8s
from datasets.cityscapes_dataset import cityscapesDataSet
import timeit
BACKBONE = 'resnet'
IGNORE_LABEL = 255
NUM_CLASSES = 19
LOG_DIR = './logs'
DATA_DIRECTORY = '/path/to/cityscapes'
DATA_LIST_PATH = './datasets/cityscapes_list/val.txt'
RESTORE_FROM = 'pretrained/'
# imageNet mean

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall',
    'fence', 'pole', 'light', 'sign',
    'vegetation', 'terrain', 'sky', 'person',
    'rider', 'car', 'truck', 'bus', 'train',
    'motocycle', 'bicycle'
]


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--backbone", type=str, default=BACKBONE,
                        help=".")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        