import os, sys
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import cv2
import json
import random

class VeriDataset(data.Dataset):
    def __init__(self, data_dir, train_list, train_data_transform = None, is_train= True ):
        super(VeriDataset, self).__init__()

        self.is_train = is_train
        self.data_dir = data_dir
        self.train_data_transform = train_data_transform
        reader = open(train_list)
        lines = reader.readlines()
        self.names = []
        self.labels = []
        self.cams = []
        if is_train == True:
            for line in lines:
                line = line.strip().split(' ')
                self.names.append(line[0])
                self.labels.append(line[1])
                self.cams.append(line[0].strip().split('_')[1])
        else:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])
        

    def __getitem__(self, index):
        # For normalize

        img = Image.open(os.path.join(self.data_dir, self.names[index])).convert('RGB') # convert gray to rgb
        target = int(self.labels[index])
        camid = self.cams[index]

        if self.train_data_transform!=None:
            img = self.train_data_transform(img)


        return img, target, camid

    def __len__(self):
        return len(self.names)
