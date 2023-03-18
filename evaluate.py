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
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--img-height", type=int, default=512)
    parser.add_argument("--img-width", type=int, default=1024)
    parser.add_argument("--log-dir", type=str, default=LOG_DIR)
    parser.add_argument("--list-path", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--is-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--split", type=str, default='val',
                        help="Whether to randomly mirror the inputs during the training.")
    return parser.parse_args()


def colorize_mask(mask):
    # mask: numpy array of the mask
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def scale_image(image, scale):
    _, _, h, w = image.size()
    scale_h = int(h*scale)
    scale_w = int(w*scale)
    image = F.interpolate(image, size=(scale_h, scale_w),
                          mode='bilinear', align_corners=True)
    return image


def predict(net, image, output_size, is_mirror=True, scales=[1]):
    interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
    outputs = []
    if is_mirror:
        # image_rev = image[:, :, :, ::-1]
        image_rev = torch.flip(image, dims=[3])
        for scale in scales:
            if scale != 1:
                image_scale = scale_image(image=image, scale=scale)
                image_rev_scale = scale_image(image=image_rev, scale=scale)
            else:
                image_scale = image
                image_rev_scale = image_rev
            image_scale = torch.cat([image_scale, image_rev_scale], dim=0)
            with torch.no_grad():
                prediction = net(image_sc