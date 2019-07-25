# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/networks/
# rm e.l && python networks.py 2>&1 | tee -a e.l && code e.l

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys
import time
import os
import copy
import glob
import cv2
import natsort 
from PIL import Image
from skimage.transform import resize
import scipy.misc

def init_weights(m):
  if type(m)==nn.Linear \
    or type(m)==nn.Conv2d \
    or type(m)==nn.ConvTranspose2d:
    torch.nn.init.xavier_uniform_(m.weight.data)
    # torch.nn.init.xavier_uniform_(m.bias)
    # m.bias.data.fill_(0.01)

class Interpolate(nn.Module):
    def __init__(
        self,size=None,scale_factor=None,mode="bilinear",align_corners=True):
        super(Interpolate,self).__init__()
        self.interp=F.interpolate
        self.size=size
        self.scale_factor=scale_factor
        self.mode=mode
        self.align_corners=align_corners

    def forward(self,x):
        x=self.interp(
            x,size=self.size,scale_factor=self.scale_factor,
            mode=self.mode,align_corners=self.align_corners)
        return x

# ================================================================================
class Pretrained_ResNet50(nn.Module):

  def __init__(self):
    super(Pretrained_ResNet50,self).__init__()
    
    # c resnet50: pretrained resnet50
    self.resnet50=models.resnet50(pretrained=True)
    # region resnet50
    # print("self.resnet50",self.resnet50)
    # endregion 
    
    # ================================================================================
    # Freeze all parameters because you will not train (or edit) those pretrained parameters
    for param in self.resnet50.parameters():
      param.requires_grad=False

    # ================================================================================
    # print("self.resnet50",dir(self.resnet50))
    self.resnet50.fc=nn.Linear(in_features=2048,out_features=1,bias=True)

    # ================================================================================
    # Check edited model
    # print("self.resnet50",self.resnet50)
    # region edited resnet50

    # endregion 

    # ================================================================================
    # self.softmax_layer=nn.Softmax(dim=None)

  def forward(self,x):
    # print("x p",x.shape)

    # ================================================================================
    x=self.resnet50(x)
    # print("x p",x.shape)

    # x=self.softmax_layer(x)

    return x