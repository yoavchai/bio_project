
import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import helper
import torchvision.utils
#import simulation
import random
import math

import os
import os.path

import numpy as np
import torch.utils.data as data
# from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from utils import *
import random
import torch
import torch.nn as nn
import math
from PIL import Image
import PIL

from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):


    def __init__(self, X, Y , test):
        self.X = X
        self.Y = Y
        self.test = test
        #self.train_crop_size = train_crop_size
        #self.test_crop_size_w = 4096#
        #self.test_crop_size_h = 3072#
        #self.round_test_images = round_test_images
        #self.crop_test = crop_test

        self.horizontal_flip = JointHorizontalFlip()
        self.vertical_flip = JointVerticalFlip()
        self.normlize = JointNormailze(means=[0.5, 0.5, 0.5], stds=[1, 1, 1])
        self.to_tensor = JointToTensor()

        self.tensor_rotate = TensorJointRotate()
        self.tensor_horizontal_flip = TensorJointHorizontalFlip()


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):

        #img , target = self.X[index], self.Y[index]
        ind = min(index, self.__len__()-2)
        ind = max(ind,1)
        # Make sure taking the pictures of same patient
        p_number = self.Y[ind].split('/')[-1].split('_')[1]
        after_p_number = self.Y[ind+1].split('/')[-1].split('_')[1]
        before_p_number = self.Y[ind - 1].split('/')[-1].split('_')[1]
        if p_number != after_p_number:
            ind -= 1
        elif p_number != before_p_number:
            ind += 1
        p_number = self.Y[ind].split('/')[-1].split('_')[1]
        after_p_number = self.Y[ind + 1].split('/')[-1].split('_')[1]
        before_p_number = self.Y[ind - 1].split('/')[-1].split('_')[1]
        assert (p_number == after_p_number)
        assert (p_number == before_p_number)
        img, target = np.asarray(PIL.Image.open(self.X[ind])) , np.asarray(PIL.Image.open(self.Y[ind]))
        try:
            img_before, img_after = np.asarray(PIL.Image.open(self.X[ind-1])), np.asarray(PIL.Image.open(self.X[ind+1]))
        except:
            print ("Index:", ind)

        img = np.transpose(img, (2, 0, 1))
        img_before, img_after = np.transpose(img_before, (2, 0, 1)), np.transpose(img_after, (2, 0, 1))
        target = np.transpose(target, (2, 0, 1))

        img, target = self.to_tensor(img, target)
        img_before, img_after = self.to_tensor(img_before, img_after)

        if (self.test == False):
            img, target,img_before, img_after = self.tensor_rotate(img,target,img_before,img_after)
            img, target,img_before, img_after = self.tensor_horizontal_flip(img,target, img_before, img_after)
            #img, target = self.horizontal_flip(img,target)
            #img, target = self.vertical_flip(img,target)



        target = target[0,:,:] #all channels should be the same
        target = torch.stack(((target == 0), (target == 127) , (target == 255))).to(dtype=torch.float32)
        #target_lesions = torch.stack(((target <= 129), (target > 129))).to(dtype=torch.float32)

        #normlize
        img = img / 255
        img_before = img_before / 255
        img_after  = img_after / 255
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        img_before = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_before)
        img_after = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img_after)
        #target = transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))(target)



        return img, target , img_before, img_after

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.Y)



# # Generate some random images
# input_images, target_masks = simulation.generate_random_data(192, 192, count=3)
#
# for x in [input_images, target_masks]:
#     print(x.shape)
#     print(x.min(), x.max())
#
# # Change channel-order and make 3 channels for matplot
# input_images_rgb = [x.astype(np.uint8) for x in input_images]
#
# # Map each channel (i.e. class) to each color
# target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]
#
# # Left: Input image, Right: Target mask (Ground-truth)
# helper.plot_side_by_side([input_images_rgb, target_masks_rgb])
#
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, datasets, models
#
#
# class SimDataset(Dataset):
#     def __init__(self, count, transform=None):
#         self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.input_images)
#
#     def __getitem__(self, idx):
#         image = self.input_images[idx]
#         mask = self.target_masks[idx]
#         if self.transform:
#             image = self.transform(image)
#
#         return [image, mask]
#
#
# # use same transform for train/val for this example
# trans = transforms.Compose([
#     transforms.ToTensor(),
# ])
#
# train_set = SimDataset(2000, transform=trans)
# val_set = SimDataset(200, transform=trans)
#
# image_datasets = {
#     'train': train_set, 'val': val_set
# }
#
# batch_size = 25
#
# dataloaders = {
#     'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
#     'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
# }
#
# dataset_sizes = {
#     x: len(image_datasets[x]) for x in image_datasets.keys()
# }
#
# dataset_sizes





def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

