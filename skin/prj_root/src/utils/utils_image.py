import matplotlib as mpl
from PIL import Image
import PIL.ImageOps
import cv2
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import image
import timeit
import sys
import scipy.misc
from tqdm import tqdm
from skimage.transform import resize
from scipy.ndimage import convolve
from skimage.restoration import denoise_tv_chambolle

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
from torchvision import transforms
from torch.autograd import Variable

# ================================================================================
# currentdir = "/mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils"
# network_dir="/mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/networks"
# sys.path.insert(0,network_dir)

currentdir = "/mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root"
sys.path.insert(0,currentdir)

# import networks as networks

from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image

# ================================================================================
def i_show_plt(img,gray=False):
    if gray==False:
        plt.imshow(img)
        plt.show()
    else:
        plt.imshow(img,cmap="gray")
        plt.show()

def resize_img(img,img_h,img_w):
    # print("img",img.shape)
    # img (348, 819, 3)
    rs_img=resize(img,(img_h,img_w))
    # print("rs_img",rs_img.shape)
    # rs_img (512, 512, 3)

    return rs_img

def load_img(path,gray=False):
    if gray==True:
        img=Image.open(path).convert("L")
        img=np.array(img)
    else:
        img=Image.open(path)
        img=np.array(img)
    return img

def compute_img_mean_std(image_paths):
    """
    computing the mean and std of three channel on the whole dataset,
    first we should normalize the image from 0-255 to 0-1
    """

    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    # ================================================================================
    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    # ================================================================================
    imgs = np.stack(imgs, axis=3)
    # print(imgs.shape)
    # (224, 224, 3, 10015)

    # ================================================================================
    imgs = imgs.astype(np.float32) / 255.

    # ================================================================================
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # ================================================================================                                                                                                                                            
    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    # ================================================================================
    print("normMean = {}".format(means))
    # normMean = [0.7630332, 0.5456441, 0.57004464]

    print("normStd = {}".format(stdevs))
    # normStd = [0.14092824, 0.15261357, 0.1699713]

    return means, stdevs

# ================================================================================
# for i in range(irg_imgs.shape[0]):
#     plt.imshow(irg_imgs[i,:,:,:].detach().cpu().numpy().transpose(1,2,0))
#     plt.show()

# for i in range(sha.shape[0]):
#     plt.imshow(sha[i,0,:,:].detach().cpu().numpy(),cmap="gray")
#     plt.show()

# for i in range(sampled_rgt_imgs.shape[0]):
#     plt.subplot(1,2,1)
#     plt.imshow(sampled_trn_imgs[i,:,:,:])
#     plt.subplot(1,2,2)
#     plt.imshow(sampled_rgt_imgs[i,:,:,:])
#     plt.show()
