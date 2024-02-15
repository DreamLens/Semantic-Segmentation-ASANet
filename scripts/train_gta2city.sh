#!/usr/bin/bash
set -ex
ls
export PYTHONPATH=`pwd`
DATA_DIR='/data/'
SNAPSHOTS_DIR='../snapshots/'
echo ${DATA_DIR}
CUDA_VISIBLE_DEVICES=0 python train_gta2city.py \
            --method='GTA5KLASA' \
            --backbone='resnet'\
            --data-dir=${DATA_DIR}gta5 \
            --data-list='./datasets/gta5_list/train.txt' \
            --data-dir-target=${DATA_DIR}cityscapes \
            --data-list-target='./datasets/cityscapes_list/train.txt' \
            --snapshot-dir=${SNAPSHOTS_DIR} \