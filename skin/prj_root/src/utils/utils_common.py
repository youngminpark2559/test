# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

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
from sklearn.feature_extraction import image
import skimage
from skimage.morphology import square
from skimage.restoration import denoise_tv_chambolle
import timeit
import sys,os
import glob
import natsort 
from itertools import zip_longest # for Python 3.x
import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
# import torchvision
# import torchvision.transforms as transforms
# from torchvision import datasets, models, transforms

# ================================================================================
# network_dir="./networks"
# sys.path.insert(0,network_dir)
# import networks as networks
from src.networks import networks as networks

# utils_dir="./utils"
# sys.path.insert(0,utils_dir)
# import utils_image as utils_image
from src.utils import utils_image as utils_image

# ================================================================================
def get_file_list(path):
    file_list=glob.glob(path)
    file_list=natsort.natsorted(file_list,reverse=False)
    return file_list

def return_path_list_from_txt(txt_file):
    txt_file=open(txt_file, "r")
    read_lines=txt_file.readlines()
    num_file=int(len(read_lines))
    txt_file.close()

    # print("read_lines",read_lines)
    # ['/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/kaggle.com_saketc_skin-lesion-analyser-sla/base_dir/train_dir/akiec/_0_133028.jpg\n', 
    #  '/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/kaggle.com_saketc_skin-lesion-analyser-sla/base_dir/train_dir/akiec/_0_1605139.jpg\n', 

    # print("num_file",num_file)
    # 31139

    return read_lines,num_file

def see_directory(path_of_dir):
    print(os.listdir(path_of_dir))
    # ['HAM10000_images_part_1.zip', 'HAM10000_images_part_2.zip', 'HAM10000_images_part_1', 'HAM10000_images_part_2', 'HAM10000_metadata.csv', 'hmnist_28_28_RGB.csv', 'hmnist_8_8_L.csv', 'hmnist_8_8_RGB.csv', 'hmnist_28_28_L.csv']

def chunks(l,n):
    # For item i in range that is length of l,
    for i in range(0,len(l),n):
        # Create index range for l of n items:
        yield l[i:i+n]

def divisorGenerator(n):
    large_divisors=[]
    for i in range(1,int(math.sqrt(n)+1)):
        if n%i==0:
            yield i
            if i*i!=n:
                large_divisors.append(n/i)
    for divisor in reversed(large_divisors):
        yield int(divisor)
# list(divisorGenerator(1024))

