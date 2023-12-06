
import argparse


METHOD = 'klasa'
BACKBONE = 'resnet'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 8
DATA_DIRECTORY = '/home/weizhou/data/gta5'
DATA_LIST_PATH = './datasets/gta5_list/train.txt'
PSEUDO_ROOT = 'results/cityscapes_pseudo_CE80000'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = '/home/weizhou/data/cityscapes'
DATA_LIST_PATH_TARGET = './datasets/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4

MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 250000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'pretrained/DeepLab_resnet_pretrained_init-f81d91e8.pth'
RESUME = ''
RESUME = False
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 2500
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
USE_WEIGHT = True
USE_SWA = True
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 1e-3
LAMBDA_ADV_TARGET2 = 2e-4
TARGET = 'cityscapes'
SET = 'train'
TEMPERATURE = 1.0


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--method", type=str, default=METHOD,
                        help="method name")
    parser.add_argument("--backbone", type=str, default=BACKBONE,
                        help="method name")
    parser.add_argument("--target", type=str, default=TARGET,