#!/bin/bash

# Path to Images
ImagePath_veri="./data/VeRi/image_train/"
TrainList_veri="./list/veri_train_list.txt"
# Number of classes
num_veri=576

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py $ImagePath_veri $TrainList_veri -n $num_veri --batch_size 64 --val_step 5 --write-out --start_epoch 0 --backbone resnet50 --save_dir './models/resnet50/' --epochs 100 > logs/resnet50.log 2>&1 &
