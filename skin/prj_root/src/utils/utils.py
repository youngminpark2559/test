# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils
# rm e.l && python utils.py 2>&1 | tee -a e.l && code e.l

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

currentdir = "/mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils"
network_dir="/mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/networks"
sys.path.insert(0,network_dir)

import networks as networks


eps = np.finfo(np.float32).eps  # threshold

# ---------------
if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"

# device=torch.device(device)
device=torch.device("cuda:0")

checkpoint_path="./direct_intrinsic_net.pth.tar"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def colorize(intensity, image, eps=1e-3):
    """
    This method reconstructs color image from reflectance intensity and input image.
    """
    # print("intensity",intensity.shape)
    # (224, 224)

    # print("intensity",intensity)

    # print("image",image.shape)
    # (224, 224, 3)

    # print("image",image)
    
    # Mean value of color channel
    # It becomes 1 channel grayscale image
    norm_input = np.mean(image, axis=2)
    # plt.imshow(norm_input,cmap="gray")
    # plt.show()

    shading = norm_input / intensity

    shading = cv2.normalize(shading, None, alpha=0.01, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    reflectance = image / np.maximum(shading, eps)[:, :, np.newaxis]

    return reflectance, shading

def normalize(img):
    """
    This method normalizes image to range of 0-1.
    """
    img = img.copy()
    
    if np.max(img) > 1:
        # img /= np.max(img)
        # # ignore some overly high pixel values in 0.1% of the image:
        img /= np.percentile(img, 99.9, interpolation='lower')
        img = np.clip(img, 0, 1)
    return img


def save_checkpoint(state, filename='./checkpoint.pth'):
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
        if args.use_saved_model=="True":
            # c loaded_parameters: loaded parameters
            loaded_parameters=torch.load(args.trained_network_path)

            # c direct_intrinsic_net: created network instance
            direct_intrinsic_net=networks.Direct_Intrinsic_Net_With_Pretrained_ResNet152().to(device)
            
            direct_intrinsic_net.load_state_dict(loaded_parameters)
        else:
            # direct_intrinsic_net=networks.Direct_Intrinsic_Net_With_Pretrained_ResNet152().to(device)
            direct_intrinsic_net=networks.Direct_Intrinsic_Net_With_Pretrained_ResNet152_Deconv().to(device)
            
            optimizer=torch.optim.Adam(
                direct_intrinsic_net.parameters(),lr=0.001)
            # optimizer=torch.optim.SGD(direct_intrinsic_net.parameters(),lr=0.01,momentum=0.9)

            print("direct_intrinsic_net.parameters()",direct_intrinsic_net.parameters())

            print_network(direct_intrinsic_net)

            return direct_intrinsic_net,optimizer
    else:
        if args.use_saved_model=="True":
            print("here")
            # direct_intrinsic_net=networks.Direct_Intrinsic_Net_based_on_paper_without_pretrained_resnet().to(device)
            direct_intrinsic_net=networks.Direct_Intrinsic_Net_tnestmeyer().to(device)

            checkpoint = torch.load(checkpoint_path)

            start_epoch = checkpoint['epoch']

            direct_intrinsic_net.load_state_dict(checkpoint['state_dict'])

            optimizer=torch.optim.Adam(
                direct_intrinsic_net.parameters(),lr=0.001)
            # optimizer=torch.optim.SGD(direct_intrinsic_net.parameters(),lr=0.01,momentum=0.9)

            optimizer.load_state_dict(checkpoint['optimizer'])

            print("direct_intrinsic_net.parameters()",direct_intrinsic_net.parameters())

            print_network(direct_intrinsic_net)

            return direct_intrinsic_net,checkpoint,optimizer
        else:
            # direct_intrinsic_net=networks.Direct_Intrinsic_Net_based_on_paper_without_pretrained_resnet().to(device)
            # direct_intrinsic_net=networks.Direct_Intrinsic_Net_With_Pretrained_ResNet152().to(device)
            direct_intrinsic_net=networks.Direct_Intrinsic_Net_tnestmeyer().to(device)
            # direct_intrinsic_net=networks.Direct_Intrinsic_Net_based_on_paper_simple_without_pretrained_resnet().to(device)
            
            direct_intrinsic_net.apply(weights_init)

            optimizer=torch.optim.Adam(
                direct_intrinsic_net.parameters(),lr=0.001)
            # optimizer=torch.optim.SGD(direct_intrinsic_net.parameters(),lr=0.01,momentum=0.9)

            print("direct_intrinsic_net.parameters()",direct_intrinsic_net.parameters())

            print_network(direct_intrinsic_net)

            return direct_intrinsic_net,optimizer

def _lightness(r):
    num_channels = len(r)
    small_v=Variable(torch.Tensor([0.0001]).cuda())
    one_vec=Variable(torch.ones(3).cuda())
    if num_channels == 3:
        L = torch.max(small_v, torch.mean(r))
        dLdR = one_vec/3.
    elif num_channels == 1:
        L = torch.max(small_v, r)
        dLdR = one_vec/1.0
    else:   
        raise Exception("Expecting 1 or 3 channels to compute lightness!")
    return L, dLdR