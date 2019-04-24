#!/bin/bash

# Path to Images
queryPath_veri="./data/VeRi/image_query/"
queryList_veri="./list/veri_query_list.txt"
galleryPath_veri="./data/VeRi/image_test/"
galleryList_veri="./list/veri_test_list.txt"


# Number of classes
num_veri=576

CUDA_VISIBLE_DEVICES=1 python -u eval.py $queryPath_veri $queryList_veri $galleryPath_veri $galleryList_veri --dataset veri --backbone resnet50 --weights './models/resnet50/Car_epoch_50.pth' --save_dir './results/veri/resnet50/'

