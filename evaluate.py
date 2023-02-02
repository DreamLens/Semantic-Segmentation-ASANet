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
DA