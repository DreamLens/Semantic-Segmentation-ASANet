set -ex
ls
export PYTHONPATH=`pwd`
CUDA_VISIBLE_DEVICES=0 python evaluate.py --backbone='resnet'\
        --data-dir='/path/to/cityscapes' \
        --list-path='./datasets/cityscapes_list/val.txt' \
        --log-dir=logs/GTA5ASA \
        --restore-from='pretrained/GTA5ASA_45.10.pth'

CUDA_VISIBLE_DEVICES=0 python evaluate.py --backbone='resnet'\
        --data-dir='/path/to/cityscapes