from PIL import Image
import PIL.ImageOps
import scipy.misc
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
import timeit
import sys

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

currentdir = "/mnt/1T-5e7/mycodehtml/cv/IID/CNN_creating_R_intensity/utils"
network_dir="/mnt/1T-5e7/mycodehtml/cv/IID/CNN_creating_R_intensity/networks"
sys.path.insert(0,network_dir)

import networks as networks

# ---------------
if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"

# device=torch.device(device)
device=torch.device("cuda:0")

def save_checkpoint(state, filename='./checkpoint.pth.tar'):
    torch.save(state, filename)

def print_network(net,struct=False):
    """
    Args
      net: created network
      struct (False): do you want to see structure of entire network?
    Print
      Structure of entire network
      Total number of parameters of network
    """
    if struct==True:
        print(net)
    
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print('Total number of parameters: %d' % num_params)

def net_generator(args):
    # print("args.use_pretrained_resnet152",args.use_pretrained_resnet152) # False

    if args.use_pretrained_resnet152=="True":
        if args.continue_training=="True":
            # c loaded_parameters: loaded parameters
            loaded_parameters=torch.load(args.trained_network_path)

            # c direct_intrinsic_net: created network instance
            direct_intrinsic_net=networks.Direct_Intrinsic_Net_With_Pretrained_ResNet152().to(device)
            
            direct_intrinsic_net.load_state_dict(loaded_parameters)
        else:
            direct_intrinsic_net=networks.Direct_Intrinsic_Net_With_Pretrained_ResNet152().to(device)
    else:
        if args.continue_training=="True":
            direct_intrinsic_net=networks.Direct_Intrinsic_Net_based_on_paper_without_pretrained_resnet().to(device)

            checkpoint = torch.load(checkpoint_path)

            start_epoch = checkpoint['epoch']

            direct_intrinsic_net.load_state_dict(checkpoint['state_dict'])

            # optimizer=torch.optim.Adam(
            #     direct_intrinsic_net.parameters(),lr=0.001)
            optimizer=torch.optim.SGD(direct_intrinsic_net.parameters(),lr=0.001,momentum=0.9)

            optimizer.load_state_dict(checkpoint['optimizer'])

            print("direct_intrinsic_net.parameters()",direct_intrinsic_net.parameters())

            print_network(direct_intrinsic_net)

            return direct_intrinsic_net,checkpoint,optimizer
        else:
            direct_intrinsic_net=networks.Direct_Intrinsic_Net_based_on_paper_without_pretrained_resnet().to(device)
            # direct_intrinsic_net=networks.Test_Net().to(device)

            # optimizer=torch.optim.Adam(
            #     direct_intrinsic_net.parameters(),lr=0.001)
            optimizer=torch.optim.SGD(direct_intrinsic_net.parameters(),lr=0.1,momentum=0.9)

            print("direct_intrinsic_net.parameters()",direct_intrinsic_net.parameters())

            print_network(direct_intrinsic_net)

            return direct_intrinsic_net,optimizer
