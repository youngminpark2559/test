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
import PIL

class Direct_Intrinsic_Net_based_on_paper_without_pretrained_resnet(nn.Module):
    def __init__(self):
        super(Direct_Intrinsic_Net_based_on_paper_without_pretrained_resnet,self).__init__()
        
        # Define layer1, 3 convolutions, 1 to 3
        self.layer1=nn.Sequential(
            # Input channel: 3, output channel: 32, kernel size: 3
            nn.Conv2d(
                in_channels=3,out_channels=32,
                kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        # Define layer2, 20 dilated convolutions, 4 to 23
        self.dilated_c1=nn.Sequential(
            nn.Conv2d(32,32,3,padding=2,dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c2=nn.Sequential(
            nn.Conv2d(32,32,3,padding=2,dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c3=nn.Sequential(
            nn.Conv2d(32,32,3,padding=2,dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c4=nn.Sequential(
            nn.Conv2d(32,32,3,padding=2,dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c5=nn.Sequential(
            nn.Conv2d(32,32,3,padding=4,dilation=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c6=nn.Sequential(
            nn.Conv2d(32,32,3,padding=4,dilation=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c7=nn.Sequential(
            nn.Conv2d(32,32,3,padding=4,dilation=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c8=nn.Sequential(
            nn.Conv2d(32,32,3,padding=4,dilation=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c9=nn.Sequential(
            nn.Conv2d(32,32,3,padding=8,dilation=8),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c10=nn.Sequential(
            nn.Conv2d(32,32,3,padding=8,dilation=8),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c11=nn.Sequential(
            nn.Conv2d(32,32,3,padding=8,dilation=8),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c12=nn.Sequential(
            nn.Conv2d(32,32,3,padding=8,dilation=8),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c13=nn.Sequential(
            nn.Conv2d(32,32,3,padding=16,dilation=16),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c14=nn.Sequential(
            nn.Conv2d(32,32,3,padding=16,dilation=16),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c15=nn.Sequential(    
            nn.Conv2d(32,32,3,padding=16,dilation=16),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c16=nn.Sequential(
            nn.Conv2d(32,32,3,padding=16,dilation=16),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c17=nn.Sequential(
            nn.Conv2d(32,32,3,padding=32,dilation=32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c18=nn.Sequential(
            nn.Conv2d(32,32,3,padding=32,dilation=32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c19=nn.Sequential(
            nn.Conv2d(32,32,3,padding=32,dilation=32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        self.dilated_c20=nn.Sequential(
            nn.Conv2d(32,32,3,padding=32,dilation=32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        # Define layer3, 3 convolutions, 24 to 26
        self.layer3=nn.Sequential(
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,1,1,padding=0))

    def forward(self,x):
        # Perform layer1, 3 convolutions
        x=self.layer1(x)
        # print("x p",x.shape) # [2, 32, 224, 224]
        # print("x p",x)
        
        # Perform 20 dilated convolutions, 10 skip connections
        o_dilated_c1=self.dilated_c1(x)
        # print("o_dilated_c1 p",o_dilated_c1.shape) # [2, 32, 224, 224]
        # print("o_dilated_c1 p",o_dilated_c1)

        o_dilated_c2=self.dilated_c2(o_dilated_c1)
        skip_c1=o_dilated_c2+x
        o_dilated_c3=self.dilated_c3(skip_c1)
        o_dilated_c4=self.dilated_c4(o_dilated_c3)
        skip_c2=o_dilated_c4+skip_c1
        # print("skip_c2 p",skip_c2.shape) # <class 'torch.Tensor'> [2, 32, 224, 224]
        # print("o_dilated_c1 p",type(skip_c2))

        o_dilated_c5=self.dilated_c5(skip_c2)
        o_dilated_c6=self.dilated_c6(o_dilated_c5)
        skip_c3=o_dilated_c6+skip_c2
        o_dilated_c7=self.dilated_c7(skip_c3)
        o_dilated_c8=self.dilated_c8(o_dilated_c7)
        skip_c4=o_dilated_c8+skip_c3
        o_dilated_c9=self.dilated_c9(skip_c4)
        o_dilated_c10=self.dilated_c10(o_dilated_c9)
        skip_c5=o_dilated_c10+skip_c4
        o_dilated_c11=self.dilated_c11(skip_c5)
        o_dilated_c12=self.dilated_c12(o_dilated_c11)
        skip_c6=o_dilated_c12+skip_c5
        o_dilated_c13=self.dilated_c13(skip_c6)
        o_dilated_c14=self.dilated_c14(o_dilated_c13)
        skip_c7=o_dilated_c14+skip_c6
        o_dilated_c15=self.dilated_c15(skip_c7)
        o_dilated_c16=self.dilated_c16(o_dilated_c15)
        skip_c8=o_dilated_c16+skip_c7
        o_dilated_c17=self.dilated_c17(skip_c8)
        o_dilated_c18=self.dilated_c18(o_dilated_c17)
        skip_c9=o_dilated_c18+skip_c8
        o_dilated_c19=self.dilated_c19(skip_c9)
        o_dilated_c20=self.dilated_c20(o_dilated_c19)
        skip_c10=o_dilated_c20+skip_c9

        # Perform layer3, 3 convolutions
        r=self.layer3(skip_c10)

        # print("r",r.shape)
        # [2, 1, 224, 224]

        # print("r p",r)

        return r


class Test_Net(nn.Module):
    def __init__(self):
        super(Test_Net,self).__init__()
        
        self.layer1=nn.Sequential(
            nn.Conv2d(
                in_channels=3,out_channels=32,
                kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,1,1,padding=0))

    def forward(self,x):
        # Perform layer1, 3 convolutions
        r=self.layer1(x)
        # print("r p",r.shape) # [2, 1, 224, 224]
        
        return r
