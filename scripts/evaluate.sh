set -ex
ls
export PYTHONPATH=`pwd`
CUDA_VISIBLE_DEVICES=0 python evaluate.py --backbone='resnet'\
        --data-dir='/path/to/cityscapes' \
        --l