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
import sys,os
import glob
from skimage.transform import resize
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable,Function
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms

# ================================================================================
# utils_dir="./utils"
# sys.path.insert(0,utils_dir)
# import utils_image as utils_image
# import utils_data as utils_data

from src.utils import utils_data as utils_data
from src.utils import utils_image as utils_image

# ================================================================================
delta=0.12  # threshold for "more or less equal"
xi=0.08  # margin in Hinge
ratio=1.0  # ratio of evaluated comparisons
eval_dense=1  # evaluate dense labels?
# m=0.425
m=0.25

# ================================================================================
def l1_loss(pred_img,gt_img):
    summed=torch.sum(torch.abs(pred_img-gt_img))
    return summed

def torch_l1_loss(pred_img,gt_img):
    criterion=nn.L1Loss()
    output=criterion(pred_img,gt_img)
    return output

def ce_loss(pred_img,gt_img):
    loss=nn.CrossEntropyLoss()
    output=loss(pred_img,gt_img)
    return output

def mse_loss(pred_img,gt_img):
    criterion=nn.MSELoss()
    output=criterion(pred_img,gt_img)
    return output

def data_lo(one_b_R_gt_imgs,pred_R_imgs,one_b_S_gt_imgs,pred_S_imgs):
    l2_r=torch.sum(torch.abs(one_b_R_gt_imgs-pred_R_imgs))
    l2_s=torch.sum(torch.abs(one_b_S_gt_imgs-pred_S_imgs))
    sum_l2=l2_r+l2_s
    return sum_l2

def data_lo_direct(one_b_R_gt_imgs,pred_R_imgs):
    l2_r=torch.sum(torch.abs(one_b_R_gt_imgs-pred_R_imgs))
    return l2_r

def grad_lo(grad_gt_R,grad_pred_R,grad_gt_S,grad_pred_S):
    l2_r=torch.sum(torch.abs(grad_gt_R-grad_pred_R))
    l2_s=torch.sum(torch.abs(grad_gt_S-grad_pred_S))
    sum_lo=l2_r+l2_s
    return sum_lo

def grad_loss_using_custom_sobel_f(dense_R_gt_img_tc,pred_R_imgs_direct):

    grad_gt_R=utils_image.image_grad(dense_R_gt_img_tc)
    grad_pred_R=utils_image.image_grad(pred_R_imgs_direct)

    loss_grad=mse_loss(grad_gt_R,grad_pred_R)
    # print("loss_grad",loss_grad)
    # loss_grad tensor(668094.5000, device='cuda:0', grad_fn=<SumBackward0>)

    return loss_grad

def grad_loss_using_subtraction(dense_R_gt_img_tc,pred_R_imgs_direct):
    grad_gt_R=utils_image.image_grad_by_subtraction(dense_R_gt_img_tc)
    grad_pred_R=utils_image.image_grad_by_subtraction(pred_R_imgs_direct)

    loss_grad=mse_loss(grad_gt_R,grad_pred_R)

    return loss_grad

def grad_lo_direct(grad_gt_R,grad_pred_R):
    l2_r=torch.sum(torch.abs(grad_gt_R-grad_pred_R))
    return l2_r

def discr_l_func_R(r_gt,r_pred):
    discr_r_loss=-torch.log(r_gt)-torch.log(1-r_pred)
    discr_r_loss[torch.isnan(discr_r_loss)]=1000.0
    return discr_r_loss

def discr_l_func_S(r_gt,s_pred):
    discr_s_loss=-torch.log(r_gt)-torch.log(1-s_pred)
    discr_s_loss[torch.isnan(discr_s_loss)]=1000.0
    return discr_s_loss

def BCELoss(data,label):
    criterion=nn.BCELoss()
    output=criterion(data,label)
    return output

def original_img_lo(one_b_O_imgs,pred_O_imgs):
    l1=torch.sum(torch.abs(one_b_O_imgs-pred_O_imgs))
    return l1

# def r_piecewise_constant_loss(
#     dense_R_gt_img_tc,dense_pred_R_imgs_direct,connected_neighbors,stride):
#     rsmooth_loss=0.0
#     # for h in range(0+50,dense_pred_R_imgs_direct.shape[2]-78,stride):
#     #     for w in range(0+50,dense_pred_R_imgs_direct.shape[3]-78,stride):
#     for h in range(260,329,stride):
#         for w in range(260,329,stride):
#             # ------------------------------------------------------------------------------------------
#             # Create matrices having center pixel and neighbor pixels
#             # c one_pix_dense_imgs: one pixel value of dense images
#             neighbor_pix_dense_R_gt_imgs=dense_R_gt_img_tc[
#                 :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]
#             # print("neighbor_pix_dense_R_gt_imgs",neighbor_pix_dense_R_gt_imgs.shape)
#             # neighbor_pix_dense_R_gt_imgs torch.Size([2, 3, 16, 16])

#             neighbor_pix_dense_R_gt_imgs_flat=neighbor_pix_dense_R_gt_imgs.reshape(
#                 neighbor_pix_dense_R_gt_imgs.shape[0],neighbor_pix_dense_R_gt_imgs.shape[1],-1)
#             # print("neighbor_pix_dense_R_gt_imgs_flat",neighbor_pix_dense_R_gt_imgs_flat.shape)
#             # neighbor_pix_dense_R_gt_imgs_flat torch.Size([2, 3, 256])

#             mean_pix_in_sub_mat=torch.mean(neighbor_pix_dense_R_gt_imgs_flat,dim=-1).unsqueeze(-1)
#             # print("mean_pix_in_sub_mat",mean_pix_in_sub_mat.shape)
#             # mean_pix_in_sub_mat torch.Size([2, 3, 1])

#             dists=torch.sum((neighbor_pix_dense_R_gt_imgs_flat-mean_pix_in_sub_mat)**2,dim=1)
#             # print("dists",dists.shape)
#             # dists torch.Size([2, 256])

#             min_dist_idx=torch.argmin(dists,dim=-1)
#             # print("min_dist_idx",min_dist_idx)
#             # min_dist_idx tensor([108, 204], device='cuda:0')

#             representitive_pix_in_sub_mat=[neighbor_pix_dense_R_gt_imgs_flat[one_img,:,min_dist_idx[one_img]] 
#                 for one_img in range(dists.shape[0])]
#             # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat)
#             # representitive_pix_in_sub_mat [tensor([0.7647, 0.7412, 0.7294], device='cuda:0'), 
#             #                                tensor([0.7463, 0.7208, 0.7054], device='cuda:0')]
#             representitive_pix_in_sub_mat=torch.stack(
#                 representitive_pix_in_sub_mat,dim=0).unsqueeze(-1).unsqueeze(-1)
#             # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat.shape)
#             # representitive_pix_in_sub_mat torch.Size([2, 3, 1, 1])
#             representitive_pix_in_sub_mat=representitive_pix_in_sub_mat.repeat(
#                 (1,1,connected_neighbors*2,connected_neighbors*2))
#             # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat.shape)
#             # representitive_pix_in_sub_mat torch.Size([2, 3, 16, 16])

#             neighbor_pix_dense_pred_R_imgs_direct=dense_pred_R_imgs_direct[
#                 :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]
#             # print("neighbor_pix_dense_pred_R_imgs_direct",neighbor_pix_dense_pred_R_imgs_direct.shape)
#             # neighbor_pix_dense_pred_R_imgs_direct torch.Size([2, 3, 16, 16])

#             dense_r_pieciwise_constant_rgb=mse_loss(
#                 neighbor_pix_dense_pred_R_imgs_direct,representitive_pix_in_sub_mat)
#             # print("dense_r_pieciwise_constant_rgb",dense_r_pieciwise_constant_rgb)
#             # dense_r_pieciwise_constant_rgb tensor(0.5500, device='cuda:0', grad_fn=<MseLossBackward>)
            
#             # one_pix_intensity_imgs=one_pix_intensity_imgs.unsqueeze(-1).unsqueeze(-1)
#             # # l2_intensity_img=one_pix_intensity_imgs-neighbor_pix_intensity_imgs
#             # mse_intensity_img=loss_functions.mse_loss(one_pix_intensity_imgs,neighbor_pix_intensity_imgs)

#             # one_pix_irg_imgs=one_pix_irg_imgs.unsqueeze(-1).unsqueeze(-1)
#             # # l2_irg_img=one_pix_irg_imgs-neighbor_pix_irg_imgs
#             # mse_irg_img=loss_functions.mse_loss(one_pix_irg_imgs,neighbor_pix_irg_imgs)

#             # summed_lo=mse_dense_img+mse_intensity_img+mse_irg_img
#             summed_lo=dense_r_pieciwise_constant_rgb
#             rsmooth_loss+=summed_lo
#     return rsmooth_loss

def r_piecewise_constant_loss(
    dense_R_gt_img_tc,dense_pred_R_imgs_direct,connected_neighbors,stride):
    rsmooth_loss=0.0
    # for h in range(0+50,dense_pred_R_imgs_direct.shape[2]-78,stride):
    #     for w in range(0+50,dense_pred_R_imgs_direct.shape[3]-78,stride):
    for h in range(0+8,dense_R_gt_img_tc.shape[2]-8+1,stride):
        for w in range(0+8,dense_R_gt_img_tc.shape[3]-8+1,stride):
            # ------------------------------------------------------------------------------------------
            # Create matrices having center pixel and neighbor pixels
            # c one_pix_dense_imgs: one pixel value of dense images
            neighbor_pix_dense_R_gt_imgs=dense_R_gt_img_tc[
                :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]
            # print("neighbor_pix_dense_R_gt_imgs",neighbor_pix_dense_R_gt_imgs.shape)
            # neighbor_pix_dense_R_gt_imgs torch.Size([2, 3, 16, 16])

            neighbor_pix_dense_R_gt_imgs_flat=neighbor_pix_dense_R_gt_imgs.reshape(
                neighbor_pix_dense_R_gt_imgs.shape[0],neighbor_pix_dense_R_gt_imgs.shape[1],-1)
            # print("neighbor_pix_dense_R_gt_imgs_flat",neighbor_pix_dense_R_gt_imgs_flat.shape)
            # neighbor_pix_dense_R_gt_imgs_flat torch.Size([2, 3, 256])

            mean_pix_in_sub_mat=torch.mean(neighbor_pix_dense_R_gt_imgs_flat,dim=-1).unsqueeze(-1)
            # print("mean_pix_in_sub_mat",mean_pix_in_sub_mat.shape)
            # mean_pix_in_sub_mat torch.Size([2, 3, 1])

            dists=torch.sum((neighbor_pix_dense_R_gt_imgs_flat-mean_pix_in_sub_mat)**2,dim=1)
            # print("dists",dists.shape)
            # dists torch.Size([2, 256])

            min_dist_idx=torch.argmin(dists,dim=-1)
            # print("min_dist_idx",min_dist_idx)
            # min_dist_idx tensor([108, 204], device='cuda:0')

            representitive_pix_in_sub_mat=[neighbor_pix_dense_R_gt_imgs_flat[one_img,:,min_dist_idx[one_img]] 
                for one_img in range(dists.shape[0])]
            # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat)
            # representitive_pix_in_sub_mat [tensor([0.7647, 0.7412, 0.7294], device='cuda:0'), 
            #                                tensor([0.7463, 0.7208, 0.7054], device='cuda:0')]
            representitive_pix_in_sub_mat=torch.stack(
                representitive_pix_in_sub_mat,dim=0).unsqueeze(-1).unsqueeze(-1)
            # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat.shape)
            # representitive_pix_in_sub_mat torch.Size([2, 3, 1, 1])
            representitive_pix_in_sub_mat=representitive_pix_in_sub_mat.repeat(
                (1,1,connected_neighbors*2,connected_neighbors*2))
            # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat.shape)
            # representitive_pix_in_sub_mat torch.Size([2, 3, 16, 16])

            neighbor_pix_dense_pred_R_imgs_direct=dense_pred_R_imgs_direct[
                :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]
            # print("neighbor_pix_dense_pred_R_imgs_direct",neighbor_pix_dense_pred_R_imgs_direct.shape)
            # neighbor_pix_dense_pred_R_imgs_direct torch.Size([2, 3, 16, 16])

            dense_r_pieciwise_constant_rgb=mse_loss(
                neighbor_pix_dense_pred_R_imgs_direct,representitive_pix_in_sub_mat)
            # print("dense_r_pieciwise_constant_rgb",dense_r_pieciwise_constant_rgb)
            # dense_r_pieciwise_constant_rgb tensor(0.5500, device='cuda:0', grad_fn=<MseLossBackward>)
            
            # one_pix_intensity_imgs=one_pix_intensity_imgs.unsqueeze(-1).unsqueeze(-1)
            # # l2_intensity_img=one_pix_intensity_imgs-neighbor_pix_intensity_imgs
            # mse_intensity_img=loss_functions.mse_loss(one_pix_intensity_imgs,neighbor_pix_intensity_imgs)

            # one_pix_irg_imgs=one_pix_irg_imgs.unsqueeze(-1).unsqueeze(-1)
            # # l2_irg_img=one_pix_irg_imgs-neighbor_pix_irg_imgs
            # mse_irg_img=loss_functions.mse_loss(one_pix_irg_imgs,neighbor_pix_irg_imgs)

            # summed_lo=mse_dense_img+mse_intensity_img+mse_irg_img
            summed_lo=dense_r_pieciwise_constant_rgb
            rsmooth_loss+=summed_lo
    return rsmooth_loss

# def r_piecewise_constant_loss_w_mask(
#     dense_R_gt_img_tc,dense_pred_R_imgs_direct,cgmit_mask_3c_imgs_tc,connected_neighbors,stride):
#     rsmooth_loss=0.0
#     # for h in range(0+50,dense_pred_R_imgs_direct.shape[2]-78,stride):
#     #     for w in range(0+50,dense_pred_R_imgs_direct.shape[3]-78,stride):
#     for h in range(260,329,stride):
#         for w in range(260,329,stride):
#             # ------------------------------------------------------------------------------------------
#             # Create matrices having center pixel and neighbor pixels
#             # c one_pix_dense_imgs: one pixel value of dense images
#             neighbor_pix_dense_R_gt_imgs=dense_R_gt_img_tc[
#                 :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]
#             # print("neighbor_pix_dense_R_gt_imgs",neighbor_pix_dense_R_gt_imgs.shape)
#             # neighbor_pix_dense_R_gt_imgs torch.Size([2, 3, 16, 16])

#             neighbor_pix_dense_R_gt_imgs_flat=neighbor_pix_dense_R_gt_imgs.reshape(
#                 neighbor_pix_dense_R_gt_imgs.shape[0],neighbor_pix_dense_R_gt_imgs.shape[1],-1)
#             # print("neighbor_pix_dense_R_gt_imgs_flat",neighbor_pix_dense_R_gt_imgs_flat.shape)
#             # neighbor_pix_dense_R_gt_imgs_flat torch.Size([2, 3, 256])

#             mean_pix_in_sub_mat=torch.mean(neighbor_pix_dense_R_gt_imgs_flat,dim=-1).unsqueeze(-1)
#             # print("mean_pix_in_sub_mat",mean_pix_in_sub_mat.shape)
#             # mean_pix_in_sub_mat torch.Size([2, 3, 1])

#             dists=torch.sum((neighbor_pix_dense_R_gt_imgs_flat-mean_pix_in_sub_mat)**2,dim=1)
#             # print("dists",dists.shape)
#             # dists torch.Size([2, 256])

#             min_dist_idx=torch.argmin(dists,dim=-1)
#             # print("min_dist_idx",min_dist_idx)
#             # min_dist_idx tensor([108, 204], device='cuda:0')

#             representitive_pix_in_sub_mat=[neighbor_pix_dense_R_gt_imgs_flat[one_img,:,min_dist_idx[one_img]] 
#                 for one_img in range(dists.shape[0])]
#             # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat)
#             # representitive_pix_in_sub_mat [tensor([0.7647, 0.7412, 0.7294], device='cuda:0'), 
#             #                                tensor([0.7463, 0.7208, 0.7054], device='cuda:0')]
#             representitive_pix_in_sub_mat=torch.stack(
#                 representitive_pix_in_sub_mat,dim=0).unsqueeze(-1).unsqueeze(-1)
#             # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat.shape)
#             # representitive_pix_in_sub_mat torch.Size([2, 3, 1, 1])
#             representitive_pix_in_sub_mat=representitive_pix_in_sub_mat.repeat(
#                 (1,1,connected_neighbors*2,connected_neighbors*2))
#             # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat.shape)
#             # representitive_pix_in_sub_mat torch.Size([2, 3, 16, 16])

#             neighbor_pix_dense_pred_R_imgs_direct=dense_pred_R_imgs_direct[
#                 :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]
#             # print("neighbor_pix_dense_pred_R_imgs_direct",neighbor_pix_dense_pred_R_imgs_direct.shape)
#             # neighbor_pix_dense_pred_R_imgs_direct torch.Size([2, 3, 16, 16])
            
#             # cgmit_mask_3c_imgs_tc_sub_mat=cgmit_mask_3c_imgs_tc[
#             #     :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]

#             # neighbor_pix_dense_pred_R_imgs_direct=torch.mul(neighbor_pix_dense_pred_R_imgs_direct,cgmit_mask_3c_imgs_tc_sub_mat)
#             # representitive_pix_in_sub_mat=torch.mul(representitive_pix_in_sub_mat,cgmit_mask_3c_imgs_tc_sub_mat)

#             # sq_term=(neighbor_pix_dense_pred_R_imgs_direct-representitive_pix_in_sub_mat)**2
#             # # mask_sq_term=torch.mul(cgmit_mask_3c_imgs_tc_sub_mat,sq_term)
#             # # print("mask_sq_term",mask_sq_term.shape)
#             # # mask_sq_term torch.Size([2, 3, 16, 16])
#             # # dense_r_pieciwise_constant_rgb=torch.sqrt(torch.sum(mask_sq_term))
#             # dense_r_pieciwise_constant_rgb=torch.sqrt(torch.sum(sq_term))
            

#             dense_r_pieciwise_constant_rgb=mse_loss(
#                 neighbor_pix_dense_pred_R_imgs_direct,representitive_pix_in_sub_mat)
#             # # print("dense_r_pieciwise_constant_rgb",dense_r_pieciwise_constant_rgb)
#             # # dense_r_pieciwise_constant_rgb tensor(0.5500, device='cuda:0', grad_fn=<MseLossBackward>)
            
#             # # one_pix_intensity_imgs=one_pix_intensity_imgs.unsqueeze(-1).unsqueeze(-1)
#             # # # l2_intensity_img=one_pix_intensity_imgs-neighbor_pix_intensity_imgs
#             # # mse_intensity_img=loss_functions.mse_loss(one_pix_intensity_imgs,neighbor_pix_intensity_imgs)

#             # # one_pix_irg_imgs=one_pix_irg_imgs.unsqueeze(-1).unsqueeze(-1)
#             # # # l2_irg_img=one_pix_irg_imgs-neighbor_pix_irg_imgs
#             # # mse_irg_img=loss_functions.mse_loss(one_pix_irg_imgs,neighbor_pix_irg_imgs)

#             # summed_lo=mse_dense_img+mse_intensity_img+mse_irg_img
#             summed_lo=dense_r_pieciwise_constant_rgb
#             rsmooth_loss+=summed_lo
#     return rsmooth_loss

def r_piecewise_constant_loss_w_mask(
    dense_R_gt_img_tc,dense_pred_R_imgs_direct,cgmit_mask_3c_imgs_tc,connected_neighbors,stride):
    rsmooth_loss=0.0
    # for h in range(0+50,dense_pred_R_imgs_direct.shape[2]-78,stride):
    #     for w in range(0+50,dense_pred_R_imgs_direct.shape[3]-78,stride):
    for h in range(0+8,dense_R_gt_img_tc.shape[2]-8+1,stride):
        for w in range(0+8,dense_R_gt_img_tc.shape[3]-8+1,stride):
            # ------------------------------------------------------------------------------------------
            # Create matrices having center pixel and neighbor pixels
            # c one_pix_dense_imgs: one pixel value of dense images
            neighbor_pix_dense_R_gt_imgs=dense_R_gt_img_tc[
                :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]
            # print("neighbor_pix_dense_R_gt_imgs",neighbor_pix_dense_R_gt_imgs.shape)
            # neighbor_pix_dense_R_gt_imgs torch.Size([2, 3, 16, 16])

            neighbor_pix_dense_R_gt_imgs_flat=neighbor_pix_dense_R_gt_imgs.reshape(
                neighbor_pix_dense_R_gt_imgs.shape[0],neighbor_pix_dense_R_gt_imgs.shape[1],-1)
            # print("neighbor_pix_dense_R_gt_imgs_flat",neighbor_pix_dense_R_gt_imgs_flat.shape)
            # neighbor_pix_dense_R_gt_imgs_flat torch.Size([2, 3, 256])

            mean_pix_in_sub_mat=torch.mean(neighbor_pix_dense_R_gt_imgs_flat,dim=-1).unsqueeze(-1)
            # print("mean_pix_in_sub_mat",mean_pix_in_sub_mat.shape)
            # mean_pix_in_sub_mat torch.Size([2, 3, 1])

            dists=torch.sum((neighbor_pix_dense_R_gt_imgs_flat-mean_pix_in_sub_mat)**2,dim=1)
            # print("dists",dists.shape)
            # dists torch.Size([2, 256])

            min_dist_idx=torch.argmin(dists,dim=-1)
            # print("min_dist_idx",min_dist_idx)
            # min_dist_idx tensor([108, 204], device='cuda:0')

            representitive_pix_in_sub_mat=[neighbor_pix_dense_R_gt_imgs_flat[one_img,:,min_dist_idx[one_img]] 
                for one_img in range(dists.shape[0])]
            # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat)
            # representitive_pix_in_sub_mat [tensor([0.7647, 0.7412, 0.7294], device='cuda:0'), 
            #                                tensor([0.7463, 0.7208, 0.7054], device='cuda:0')]
            representitive_pix_in_sub_mat=torch.stack(
                representitive_pix_in_sub_mat,dim=0).unsqueeze(-1).unsqueeze(-1)
            # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat.shape)
            # representitive_pix_in_sub_mat torch.Size([2, 3, 1, 1])
            representitive_pix_in_sub_mat=representitive_pix_in_sub_mat.repeat(
                (1,1,connected_neighbors*2,connected_neighbors*2))
            # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat.shape)
            # representitive_pix_in_sub_mat torch.Size([2, 3, 16, 16])

            neighbor_pix_dense_pred_R_imgs_direct=dense_pred_R_imgs_direct[
                :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]
            # print("neighbor_pix_dense_pred_R_imgs_direct",neighbor_pix_dense_pred_R_imgs_direct.shape)
            # neighbor_pix_dense_pred_R_imgs_direct torch.Size([2, 3, 16, 16])
            
            # cgmit_mask_3c_imgs_tc_sub_mat=cgmit_mask_3c_imgs_tc[
            #     :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]

            # neighbor_pix_dense_pred_R_imgs_direct=torch.mul(neighbor_pix_dense_pred_R_imgs_direct,cgmit_mask_3c_imgs_tc_sub_mat)
            # representitive_pix_in_sub_mat=torch.mul(representitive_pix_in_sub_mat,cgmit_mask_3c_imgs_tc_sub_mat)

            # sq_term=(neighbor_pix_dense_pred_R_imgs_direct-representitive_pix_in_sub_mat)**2
            # # mask_sq_term=torch.mul(cgmit_mask_3c_imgs_tc_sub_mat,sq_term)
            # # print("mask_sq_term",mask_sq_term.shape)
            # # mask_sq_term torch.Size([2, 3, 16, 16])
            # # dense_r_pieciwise_constant_rgb=torch.sqrt(torch.sum(mask_sq_term))
            # dense_r_pieciwise_constant_rgb=torch.sqrt(torch.sum(sq_term))
            

            dense_r_pieciwise_constant_rgb=mse_loss(
                neighbor_pix_dense_pred_R_imgs_direct,representitive_pix_in_sub_mat)
            # # print("dense_r_pieciwise_constant_rgb",dense_r_pieciwise_constant_rgb)
            # # dense_r_pieciwise_constant_rgb tensor(0.5500, device='cuda:0', grad_fn=<MseLossBackward>)
            
            # # one_pix_intensity_imgs=one_pix_intensity_imgs.unsqueeze(-1).unsqueeze(-1)
            # # # l2_intensity_img=one_pix_intensity_imgs-neighbor_pix_intensity_imgs
            # # mse_intensity_img=loss_functions.mse_loss(one_pix_intensity_imgs,neighbor_pix_intensity_imgs)

            # # one_pix_irg_imgs=one_pix_irg_imgs.unsqueeze(-1).unsqueeze(-1)
            # # # l2_irg_img=one_pix_irg_imgs-neighbor_pix_irg_imgs
            # # mse_irg_img=loss_functions.mse_loss(one_pix_irg_imgs,neighbor_pix_irg_imgs)

            # summed_lo=mse_dense_img+mse_intensity_img+mse_irg_img
            summed_lo=dense_r_pieciwise_constant_rgb
            rsmooth_loss+=summed_lo
    return rsmooth_loss

def r_piecewise_constant_loss_for_all_datasets(
    dense_R_gt_img_tc,dense_pred_R_imgs_direct,
    cgmit_gt_R_3c_imgs_tc,pred_cgmit_r_imgs_direct,cgmit_mask_3c_imgs_tc,
    connected_neighbors=8,stride=2):
    rsmooth_loss=0.0
    for h in range(0+connected_neighbors,dense_pred_R_imgs_direct.shape[2]-connected_neighbors,stride):
        for w in range(0+connected_neighbors,dense_pred_R_imgs_direct.shape[3]-connected_neighbors,stride):
            # ------------------------------------------------------------------------------------------
            # Create matrices having center pixel and neighbor pixels
            # c one_pix_dense_imgs: one pixel value of dense images
            neighbor_pix_dense_R_gt_imgs=dense_R_gt_img_tc[
                :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]
            # print("neighbor_pix_dense_R_gt_imgs",neighbor_pix_dense_R_gt_imgs.shape)
            # neighbor_pix_dense_R_gt_imgs torch.Size([2, 3, 16, 16])

            neighbor_pix_dense_R_gt_imgs_flat=neighbor_pix_dense_R_gt_imgs.reshape(
                neighbor_pix_dense_R_gt_imgs.shape[0],neighbor_pix_dense_R_gt_imgs.shape[1],-1)
            # print("neighbor_pix_dense_R_gt_imgs_flat",neighbor_pix_dense_R_gt_imgs_flat.shape)
            # neighbor_pix_dense_R_gt_imgs_flat torch.Size([2, 3, 256])

            mean_pix_in_sub_mat=torch.mean(neighbor_pix_dense_R_gt_imgs_flat,dim=-1).unsqueeze(-1)
            # print("mean_pix_in_sub_mat",mean_pix_in_sub_mat.shape)
            # mean_pix_in_sub_mat torch.Size([2, 3, 1])

            dists=torch.sum((neighbor_pix_dense_R_gt_imgs_flat-mean_pix_in_sub_mat)**2,dim=1)
            # print("dists",dists.shape)
            # dists torch.Size([2, 256])

            min_dist_idx=torch.argmin(dists,dim=-1)
            # print("min_dist_idx",min_dist_idx)
            # min_dist_idx tensor([108, 204], device='cuda:0')

            representitive_pix_in_sub_mat=[neighbor_pix_dense_R_gt_imgs_flat[one_img,:,min_dist_idx[one_img]] 
                for one_img in range(dists.shape[0])]
            # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat)
            # representitive_pix_in_sub_mat [tensor([0.7647, 0.7412, 0.7294], device='cuda:0'), 
            #                                tensor([0.7463, 0.7208, 0.7054], device='cuda:0')]
            representitive_pix_in_sub_mat=torch.stack(
                representitive_pix_in_sub_mat,dim=0).unsqueeze(-1).unsqueeze(-1)
            # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat.shape)
            # representitive_pix_in_sub_mat torch.Size([2, 3, 1, 1])
            representitive_pix_in_sub_mat=representitive_pix_in_sub_mat.repeat(
                (1,1,connected_neighbors*2,connected_neighbors*2))
            # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat.shape)
            # representitive_pix_in_sub_mat torch.Size([2, 3, 16, 16])

            neighbor_pix_dense_pred_R_imgs_direct=dense_pred_R_imgs_direct[
                :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]
            # print("neighbor_pix_dense_pred_R_imgs_direct",neighbor_pix_dense_pred_R_imgs_direct.shape)
            # neighbor_pix_dense_pred_R_imgs_direct torch.Size([2, 3, 16, 16])

            dense_r_pieciwise_constant_rgb=mse_loss(
                neighbor_pix_dense_pred_R_imgs_direct,representitive_pix_in_sub_mat)
            rsmooth_loss+=dense_r_pieciwise_constant_rgb

            # ------------------------------------------------------------------------------------------
            # Create matrices having center pixel and neighbor pixels
            # c one_pix_dense_imgs: one pixel value of dense images
            neighbor_pix_cgmit_R_gt_imgs=cgmit_gt_R_3c_imgs_tc[
                :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]
            # print("neighbor_pix_cgmit_R_gt_imgs",neighbor_pix_cgmit_R_gt_imgs.shape)
            # neighbor_pix_cgmit_R_gt_imgs torch.Size([2, 3, 16, 16])

            neighbor_pix_cgmit_R_gt_imgs_flat=neighbor_pix_cgmit_R_gt_imgs.reshape(
                neighbor_pix_cgmit_R_gt_imgs.shape[0],neighbor_pix_cgmit_R_gt_imgs.shape[1],-1)
            # print("neighbor_pix_cgmit_R_gt_imgs_flat",neighbor_pix_cgmit_R_gt_imgs_flat.shape)
            # neighbor_pix_cgmit_R_gt_imgs_flat torch.Size([2, 3, 256])

            mean_pix_in_sub_mat=torch.mean(neighbor_pix_cgmit_R_gt_imgs_flat,dim=-1).unsqueeze(-1)
            # print("mean_pix_in_sub_mat",mean_pix_in_sub_mat.shape)
            # mean_pix_in_sub_mat torch.Size([2, 3, 1])

            dists=torch.sum((neighbor_pix_cgmit_R_gt_imgs_flat-mean_pix_in_sub_mat)**2,dim=1)
            # print("dists",dists.shape)
            # dists torch.Size([2, 256])

            min_dist_idx=torch.argmin(dists,dim=-1)
            # print("min_dist_idx",min_dist_idx)
            # min_dist_idx tensor([108, 204], device='cuda:0')

            representitive_pix_in_sub_mat=[neighbor_pix_cgmit_R_gt_imgs_flat[one_img,:,min_dist_idx[one_img]] 
                for one_img in range(dists.shape[0])]
            # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat)
            # representitive_pix_in_sub_mat [tensor([0.7647, 0.7412, 0.7294], device='cuda:0'), 
            #                                tensor([0.7463, 0.7208, 0.7054], device='cuda:0')]
            representitive_pix_in_sub_mat=torch.stack(
                representitive_pix_in_sub_mat,dim=0).unsqueeze(-1).unsqueeze(-1)
            # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat.shape)
            # representitive_pix_in_sub_mat torch.Size([2, 3, 1, 1])
            representitive_pix_in_sub_mat=representitive_pix_in_sub_mat.repeat(
                (1,1,connected_neighbors*2,connected_neighbors*2))
            # print("representitive_pix_in_sub_mat",representitive_pix_in_sub_mat.shape)
            # representitive_pix_in_sub_mat torch.Size([2, 3, 16, 16])

            neighbor_pix_cgmit_pred_R_imgs_direct=pred_cgmit_r_imgs_direct[
                :,:,h-connected_neighbors:h+connected_neighbors,w-connected_neighbors:w+connected_neighbors]
    
            cgmit_r_pieciwise_constant_rgb=mse_loss(
                neighbor_pix_cgmit_pred_R_imgs_direct,representitive_pix_in_sub_mat)
            rsmooth_loss+=cgmit_r_pieciwise_constant_rgb
    return rsmooth_loss

class HumanReflectanceJudgements(object):

    def __init__(self, judgements):
        if not isinstance(judgements, dict):
            raise ValueError("Invalid judgements: %s" % judgements)

        self.judgements=judgements
        self.id_to_points={p['id']: p for p in self.points}

    @staticmethod
    def from_file(filename):
        judgements=json.load(open(filename))
        # c hrj: human reflectance judgement json file
        hrj=HumanReflectanceJudgements(judgements)
        return hrj

    @property
    def points(self):
        return self.judgements['intrinsic_points']

    @property
    def comparisons(self):
        return self.judgements['intrinsic_comparisons']

# region original iiw loss
def eval_single_comparison_with_SVM_hinge_loss(
  o_p_ref,o_iiw_ori_img,comparisons,hrjf,one_p_iiw_ds,applied_DA,one_iiw_img_after_DA_before_rs):

  # print("o_p_ref",o_p_ref.shape)
  # o_p_ref torch.Size([1, 1, 417, 559])

  # c o_p_ref: one predicted reflectance intensity image
  # comparisons
  # hrjf
  # c o_ori_img: one orignal image which is not processed
  # c one_p_iiw_ds: one pair of iiw dataset
  # applied_DA

  # --------------------------------------------------------------------------------
  # print("one_p_iiw_ds",one_p_iiw_ds)
  # ('/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data/iiw_trn/iiw_001/000141.png\n', 
  #  '/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data/iiw_json/iiw_001/000141.json\n')
  # /mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data/iiw_trn_lr/iiw_001/54_lr.png
  # /mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data/iiw_trn_p3/iiw_001/000054_p3.png

  # --------------------------------------------------------------------------------
  # # c one_img_of_iiw_ds: one image of iiw dataset
  # one_img_of_iiw_ds=one_p_iiw_ds[1]
  # one_img_of_iiw_ds="/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data/iiw_trn_lr/iiw_001/54_lr.png"
  # try:
  #     # c kind_of_DA: kind of data augmentation
  #     # c kind_of_DA: no_DA,lr,ud,p3,p6,p9,n3,n6,n9
  #     kind_of_DA=one_img_of_iiw_ds.split("/")[-1].split(".")[0].split("_")[-1]
  # except:
  #     import traceback
  #     print(traceback.format_exc())
  #     kind_of_DA="no_DA"
  # print("kind_of_DA",kind_of_DA)
  # kind_of_DA no_DA
  # kind_of_DA lr

  # --------------------------------------------------------------------------------
  # c refl_img: mean of one predicted reflectance
  # refl_img=torch.mean(o_p_ref,dim=1,keepdim=True).squeeze()
  # print("refl_img",refl_img.shape)
  # refl_img torch.Size([340, 512])

  # print("refl_img",refl_img.grad_fn)

  refl_img=o_p_ref.squeeze()
  # print("refl_img",refl_img.shape)
  # refl_img torch.Size([340, 512])

  # --------------------------------------------------------------------------------
  rows=o_iiw_ori_img.shape[0]
  cols=o_iiw_ori_img.shape[1]
  # print("rows",rows)
  # print("cols",cols)
  # rows 341
  # cols 512

  # --------------------------------------------------------------------------------
  # c error_sum: sum all errors from all comparisons in one image
  error_sum=0.0
  # c weight_sum: sum all weights from all comparisons in one image
  weight_sum=0.0

  # --------------------------------------------------------------------------------
  num_comp=float(len(comparisons))
  # print("num_comp",num_comp)
  # 125.0

  # --------------------------------------------------------------------------------
  # JSON GT for 1 image, 
  # containing all relative reflectance comparisons information
  # c c: one comparison from 1 image's all comparisons
  for c in comparisons:
    # c n_po1: number of point1
    n_po1=c['point1']
    # c point1: Point1 from one comparison
    point1 = hrjf.id_to_points[n_po1]

    n_po2=c['point2']
    # c point2: Point2 from one comparison
    point2=hrjf.id_to_points[n_po2]

    # c darker: Darker information from one comparison
    darker=c['darker']
    
    # Weight information from one comparison
    weight=float(c['darker_score']) # 1.14812035203497
    # print("weight",weight)

    # --------------------------------------------------------------------------------
    # Check exception
    if not point1['opaque'] or not point2['opaque']:
        # Pass this judgement
        continue
    # weight<0 or weight is None -> invalid darker_score so pass
    if weight<0 or weight is None:
        raise ValueError("Invalid darker_score: %s"%weight)
    if darker not in ('1','2','E'):
        raise ValueError("Invalid darker: %s"%darker)
    
    # ================================================================================
    x1,y1,x2,y2,darker=int(point1['x']*cols),int(point1['y']*rows),\
                       int(point2['x']*cols),int(point2['y']*rows),darker
    # print("x1,y1,x2,y2",x1,y1,x2,y2)

    pairs=[[x1,y1],[x2,y2]]

    # ================================================================================
    kind_of_DA=applied_DA
    # print("kind_of_DA",kind_of_DA)
    # kind_of_DA n3

    if kind_of_DA=="lr":
        x1=abs(x1-cols)
        x2=abs(x2-cols)
    elif kind_of_DA=="ud":
        y1=abs(y1-rows)
        y2=abs(y2-rows)
    # elif kind_of_DA=="p3":
    #     pairs_after_rot=utils_data.get_rotated_coordinates(pairs,o_iiw_ori_img,angle=3)
    #     x1=pairs_after_rot[0][0]
    #     y1=pairs_after_rot[0][1]
    #     x2=pairs_after_rot[1][0]
    #     y2=pairs_after_rot[1][1]
    # elif kind_of_DA=="p6":
    #     pairs_after_rot=utils_data.get_rotated_coordinates(pairs,o_iiw_ori_img,angle=6)
    #     x1=pairs_after_rot[0][0]
    #     y1=pairs_after_rot[0][1]
    #     x2=pairs_after_rot[1][0]
    #     y2=pairs_after_rot[1][1]
    # elif kind_of_DA=="p9":
    #     pairs_after_rot=utils_data.get_rotated_coordinates(pairs,o_iiw_ori_img,angle=9)
    #     x1=pairs_after_rot[0][0]
    #     y1=pairs_after_rot[0][1]
    #     x2=pairs_after_rot[1][0]
    #     y2=pairs_after_rot[1][1]
    # elif kind_of_DA=="n3":
    #     pairs_after_rot=utils_data.get_rotated_coordinates(pairs,o_iiw_ori_img,angle=-3)
    #     x1=pairs_after_rot[0][0]
    #     y1=pairs_after_rot[0][1]
    #     x2=pairs_after_rot[1][0]
    #     y2=pairs_after_rot[1][1]
    # elif kind_of_DA=="n6":
    #     pairs_after_rot=utils_data.get_rotated_coordinates(pairs,o_iiw_ori_img,angle=-6)
    #     x1=pairs_after_rot[0][0]
    #     y1=pairs_after_rot[0][1]
    #     x2=pairs_after_rot[1][0]
    #     y2=pairs_after_rot[1][1]
    # elif kind_of_DA=="n9":
    #     pairs_after_rot=utils_data.get_rotated_coordinates(pairs,o_iiw_ori_img,angle=-9)
    #     x1=pairs_after_rot[0][0]
    #     y1=pairs_after_rot[0][1]
    #     x2=pairs_after_rot[1][0]
    #     y2=pairs_after_rot[1][1]
    else:
        pass

    # ================================================================================
    # print("x1,y1,x2,y2",x1,y1,x2,y2)
    # x1,y1,x2,y2 362 283 402 327
    # x1,y1,x2,y2 376 290 419 332

    # c R1: scalar value of point1 from predicted intensity image
    R1=refl_img[y1,x1]
    R2=refl_img[y2,x2]
    
    # ================================================================================
    # same_cuda=torch.device('cuda:'+str(R1.get_device()))
    # R2=torch.where(R2<1e-4,torch.Tensor([1e-4]).squeeze().cuda(same_cuda),R2)
    # R2=torch.where(R2<1e-4,torch.Tensor([1e-4]).squeeze().cuda(),R2)
    # R2=torch.where(R2<1e-4,torch.cuda.FloatTensor([1e-4]),R2)
    # R2=R2+1e-4
    R2=torch.where(torch.abs(R2)<1e-4,torch.tensor([1e-4],device=R2.device),R2)
    
    # R2[torch.abs(R2)<1e-4]=1e-4 # In place error on gradient

    # ================================================================================
    div_R1_R2=torch.div(R1,R2)
    
    # c dx_inv: 1+delta+xi inverse
    dx_inv=(1.0/(1.0+delta+xi))
    
    # c dx: 1+delta+xi
    dx=(1.0+delta+xi)
    
    # c dx_m_inv: 1+delta-i inverse
    dx_m_inv=(1.0/(1.0+delta-xi))
    
    # c dx_m: 1+delta-xi
    dx_m=(1.0+delta-xi)

    zero=torch.tensor([0.0],device=div_R1_R2.device)

    # ================================================================================
    if darker=='1':
      # c ersp: error of single pair
      # ersp=torch.max(torch.Tensor([0.0]).cuda(same_cuda),div_R1_R2-dx_inv)
      # ersp=torch.max(torch.Tensor([0.0]).cuda(),div_R1_R2-dx_inv)
      ersp=torch.max(zero,div_R1_R2-dx_inv)
      # print("ersp",ersp)

      # Sum all errors from all comparisons in one image
      error_sum+=ersp

      # Sum all weights from all comparisons in one image
      weight_sum+=weight

    elif darker=='2':
      # ersp=torch.max(torch.Tensor([0.0]).cuda(same_cuda),dx-div_R1_R2)
      # ersp=torch.max(torch.Tensor([0.0]).cuda(),dx-div_R1_R2)
      ersp=torch.max(zero,dx-div_R1_R2)
      # print("ersp 2",ersp.grad_fn)
      # print("ersp 2",ersp)

      # Sum all errors from all comparisons in one image
      error_sum+=ersp

      # Sum all weights from all comparisons in one image
      weight_sum+=weight

    else:
      if xi<=delta:
        # ersp=torch.max(torch.Tensor([0.0]).cuda(same_cuda),dx_m_inv-div_R1_R2)
        # ersp=torch.max(torch.Tensor([0.0]).cuda(),dx_m_inv-div_R1_R2)
        ersp=torch.max(zero,dx_m_inv-div_R1_R2)

        # Sum all errors from all comparisons in one image
        error_sum+=ersp

        # Sum all weights from all comparisons in one image
        weight_sum+=weight

      else:
        # ersp=torch.max(torch.Tensor([0.0]).cuda(same_cuda),div_R1_R2-dx_m)
        # ersp=torch.max(torch.Tensor([0.0]).cuda(),div_R1_R2-dx_m)
        ersp=torch.max(zero,div_R1_R2-dx_m)

        # Sum all errors from all comparisons in one image
        error_sum+=ersp

        # Sum all weights from all comparisons in one image
        weight_sum+=weight

    # elif darker=='E':
    #     first_max=torch.max(div_R1_R2-dx_m,dx_m_inv-div_R1_R2)
    #     ersp=torch.max(torch.Tensor([0.0]).cuda(),first_max)
    #     # print("ersp 3",ersp.grad_fn)
    #     # print("ersp 3 1",ersp)

  weight_sum=weight_sum/num_comp
  # print("weight_sum",weight_sum)
  # 0.6306647204908248

  error_sum=error_sum/num_comp
  # print("error_sum",error_sum)
  # tensor([0.2022], device='cuda:0', grad_fn=<DivBackward0>)

  # ================================================================================
  # Now, you have processed all comparisons in one image
  # If weight_sum exist
  if weight_sum:
    # c whdr: calculated whdr of one image
    whdr=error_sum/weight_sum
    # print("whdr",whdr)
    # tensor([0.1147], device='cuda:0', grad_fn=<DivBackward0>)

  # ================================================================================
  # If weight_sum=0, it means there's no comparisons
  # In that case, you assign 0 into whdr
  else:
    whdr=0.0

  # ================================================================================
  # Return whdr score of one image
  return whdr

# endregion original iiw loss

# region original iiw loss, with clone
# def eval_single_comparison_with_SVM_hinge_loss(o_p_ref,comparisons,hrjf,o_ori_img):
#     refl_img=np.mean(o_p_ref,axis=1).squeeze()

#     # --------------------------------------------------------------------------------
#     rows=o_ori_img.shape[0]
#     cols=o_ori_img.shape[1]

#     # --------------------------------------------------------------------------------
#     # c error_sum: sum all errors from all comparisons in one image
#     error_sum=0.0
#     # c weight_sum: sum all weights from all comparisons in one image
#     weight_sum=0.0

#     # --------------------------------------------------------------------------------
#     # JSON GT for 1 image, 
#     # containing all relative reflectance comparisons information
#     # c c: one comparison from 1 image's all comparisons
#     for c in comparisons:
#         # c n_po1: number of point1
#         n_po1=c['point1']
#         # c point1: Point1 from one comparison
#         point1 = hrjf.id_to_points[n_po1]

#         n_po2=c['point2']
#         # c point2: Point2 from one comparison
#         point2=hrjf.id_to_points[n_po2]

#         # c darker: Darker information from one comparison
#         darker=c['darker']
        
#         # Weight information from one comparison
#         weight=c['darker_score']

#         # --------------------------------------------------------------------------------
#         # Check exception
#         if not point1['opaque'] or not point2['opaque']:
#             # Pass this judgement
#             continue
#         # weight<0 or weight is None -> invalid darker_score so pass
#         if weight<0 or weight is None:
#             raise ValueError("Invalid darker_score: %s" % weight)
#         if darker not in ('1','2','E'):
#             raise ValueError("Invalid darker: %s" % darker)
        
#         # --------------------------------------------------------------------------------
#         x1,y1,x2,y2,darker=int(point1['x']*cols),int(point1['y']*rows),int(point2['x']*cols),int(point2['y']*rows),darker

#         # --------------------------------------------------------------------------------        
#         # c R1: scalar value of point1 from predicted intensity image
#         # refl_img (1, 341, 512)
#         R1=refl_img[y1,x1]
#         R2=refl_img[y2,x2]
        
#         # --------------------------------------------------------------------------------
#         div_R1_R2=R1/R2
        
#         # c dx_inv: 1+delta+xi inverse
#         dx_inv=(1.0/(1.0+delta+xi))
#         # c dx: 1+delta+xi
#         dx=(1.0+delta+xi)
#         # c dx_m_inv: 1+delta-i inverse
#         dx_m_inv=(1.0/(1.0+delta-xi))
#         # c dx_m: 1+delta-xi
#         dx_m=(1.0+delta-xi)

#         # --------------------------------------------------------------------------------
#         if darker=='1':
#             # c ersp: error of single pair
#             ersp=max(0.0,div_R1_R2-dx_inv)
#             # print("ersp 1",ersp)
#         elif darker=='2':
#             ersp=max(0.0,dx-div_R1_R2)
#             # print("ersp 2",ersp)
#         # elif darker=='E':
#         #     if xi<=delta:
#         #         ersp=max(0.0,dx_m_inv-div_R1_R2)
#         #         print("ersp 3 1",ersp)
#         #     else:
#         #         ersp=max(0.0,div_R1_R2-dx_m)
#         #         print("ersp 3 2",ersp)
#         elif darker=='E':
#             first_max=max(div_R1_R2-dx_m,dx_m_inv-div_R1_R2)
#             ersp=max(0.0,first_max)
#             # print("ersp 3 1",ersp)
        
#         # Sum all errors from all comparisons in one image
#         error_sum+=ersp
#         # print("error_sum",error_sum)

#         # Sum all weights from all comparisons in one image
#         weight_sum+=weight
#         # print("weight_sum",weight_sum)

#     # Now, you have processed all comparisons in one image
#     # If weight_sum exist
#     if weight_sum:
#         # c whdr: calculated whdr of one image
#         whdr=error_sum/weight_sum
#         # print("whdr 1",whdr)

#     # If weight_sum=0, it means there's no comparisons
#     # In that case, you assign 0 into whdr
#     else:
#         whdr=0.0
#         # print("whdr 2",whdr)

#     # Return whdr score of one image
#     return whdr
# endregion original iiw loss, with clone

# region original iiw loss, updated with log
# def eval_single_comparison_with_SVM_hinge_loss(o_p_ref,comparisons,hrjf,o_ori_img):
#     # c refl_img: mean of one predicted reflectance
#     # refl_img=torch.mean(o_p_ref,dim=1,keepdim=True).squeeze()
#     # print("m_o_p_ref",m_o_p_ref.shape)
#     # m_o_p_ref torch.Size([341, 512])

#     # --------------------------------------------------------------------------------
#     refl_img=o_p_ref.squeeze()
#     # print("o_p_ref",o_p_ref.shape)
#     # o_p_ref torch.Size([341, 512])

#     rows=o_ori_img.shape[0]
#     cols=o_ori_img.shape[1]
#     # print("rows",rows)
#     # print("cols",cols)
#     # rows 341
#     # cols 512

#     # --------------------------------------------------------------------------------
#     # c error_sum: sum all errors from all comparisons in one image
#     # error_sum=0.0

#     # --------------------------------------------------------------------------------
#     num_valid_comparisons=0.0

#     num_valid_comparisons_ineq=0.0
#     num_valid_comparisons_eq=0.0

#     total_loss_eq=torch.cuda.FloatTensor(1)
#     total_loss_eq[0]=0.0
#     total_loss_ineq=torch.cuda.FloatTensor(1)
#     total_loss_ineq[0]=0.0

#     # --------------------------------------------------------------------------------
#     # JSON GT for 1 image, 
#     # containing all relative reflectance comparisons information
#     # c c: one comparison from 1 image's all comparisons
#     for c in comparisons:
#         # c n_po1: number of point1
#         n_po1=c['point1']
#         # c point1: Point1 from one comparison
#         point1 = hrjf.id_to_points[n_po1]

#         n_po2=c['point2']
#         # c point2: Point2 from one comparison
#         point2=hrjf.id_to_points[n_po2]

#         # c darker: Darker information from one comparison
#         darker=c['darker']
        
#         # Weight information from one comparison
#         weight=c['darker_score'] # 1.14812035203497
#         weight=float(weight)

#         # --------------------------------------------------------------------------------
#         # Check exception
#         if not point1['opaque'] or not point2['opaque']:
#             # Pass this judgement
#             continue
#         # weight<0 or weight is None -> invalid darker_score so pass
#         if weight<0 or weight is None:
#             raise ValueError("Invalid darker_score: %s" % weight)
#         if darker not in ('1','2','E'):
#             raise ValueError("Invalid darker: %s" % darker)
        
#         # --------------------------------------------------------------------------------
#         x1,y1,x2,y2,darker=int(point1['x']*cols),int(point1['y']*rows),int(point2['x']*cols),int(point2['y']*rows),darker

#         # --------------------------------------------------------------------------------        
#         # c R1: scalar value of point1 from predicted intensity image
#         # R1=(torch.log(refl_img[y1,x1]))
#         # R2=(torch.log(refl_img[y2,x2]))
#         R1=refl_img[y1,x1]
#         R2=refl_img[y2,x2]

#         # # --------------------------------------------------------------------------------
#         # # +1
#         # if darker=='1':
#         #     # c ersp: error of single pair
#         #     ersp=weight*(torch.max(torch.Tensor([0.0]).cuda(),m-R1+R2)**2)
#         # # -1
#         # elif darker=='2':
#         #     ersp=weight*(torch.max(torch.Tensor([0.0]).cuda(),m-R2+R1)**2)
#         # # 0
#         # elif darker=='E':
#         #     ersp=weight*((R1-R2)**2)
        
#         # # print("ersp",ersp)
#         # # print("ersp",ersp.grad_fn)
#         # # Sum all errors from all comparisons in one image
#         # # error_sum+=ersp
#         # # print("error_sum",error_sum)
#         # # print("error_sum",error_sum.grad_fn)

#         # --------------------------------------------------------------------------------
#         if darker=='1':
#             # c ersp: error of single pair
#             total_loss_ineq+=weight*torch.mean(torch.pow(m-(R2-R1),2))
#             num_valid_comparisons_ineq+=1.
#         elif darker=='2':
#             total_loss_ineq+=weight*torch.mean(torch.pow(m-(R1-R2),2))
#             num_valid_comparisons_ineq+=1.
#         elif darker=='E':
#             total_loss_eq+=weight*torch.mean(torch.pow(R1-R2,2))
#             num_valid_comparisons_eq+=1.
    
#     # --------------------------------------------------------------------------------
#     total_loss=total_loss_ineq+total_loss_eq
#     num_valid_comparisons=num_valid_comparisons_eq+num_valid_comparisons_ineq

#     total_loss_real=total_loss/(num_valid_comparisons+1e-6)
#     # print("total_loss_real",total_loss_real)

#     return total_loss_real
# endregion original iiw loss, updated with log

# region original iiw loss, updated with log, update one image at once
# def eval_single_comparison_with_SVM_hinge_loss(one_iiw_pred,one_iiw_gt):
    
#     # c refl_img: mean of one predicted reflectance
#     refl_img=torch.mean(one_iiw_pred,dim=1,keepdim=True).squeeze()
#     # print("refl_img",refl_img.shape)
#     # refl_img torch.Size([340, 512])

#     # --------------------------------------------------------------------------------
#     rows=refl_img.shape[0]
#     cols=refl_img.shape[1]

#     # --------------------------------------------------------------------------------
#     # eq_loss=0.0
#     # ineq_loss=0.0
#     # num_valid_eq=0.0
#     # num_valid_ineq=0.0
#     # final_lo=0.0
#     tau=0.425

#     # --------------------------------------------------------------------------------
#     # print("one_iiw_gt",one_iiw_gt)
#     # one_iiw_gt [[[0.56688086 0.64213177 0.65221204 0.70249289 2.         1.01427991]]
#     # print("one_iiw_gt",one_iiw_gt.shape)
#     # one_iiw_gt (1181, 6)

#     # start=timeit.default_timer()
#     # mask = ((r["dt"] >= startdate) & (r["dt"] <= enddate))
#     # (one_iiw_gt==2.)
#     ineq_mat=one_iiw_gt[np.logical_or(one_iiw_gt[:,4]==2.,one_iiw_gt[:,4]==1.)]
#     eq_mat=one_iiw_gt[one_iiw_gt[:,4]==0.]
#     # print("ineq_mat",ineq_mat.shape)
#     # ineq_mat (25, 6)
#     # print("eq_mat",eq_mat.shape)
#     # eq_mat (36, 6)
#     # stop=timeit.default_timer()
#     # print("process time:",stop-start)

#     # print("ineq_mat",ineq_mat)
#     # print("eq_mat",eq_mat)

#     # comparisons_blob[c, 0] = x1
#     # comparisons_blob[c, 1] = y1
#     # comparisons_blob[c, 2] = x2
#     # comparisons_blob[c, 3] = y2
#     # comparisons_blob[c, 4] = darker
#     # comparisons_blob[c, 5] = weight

#     if ineq_mat.shape[0]!=0 and eq_mat.shape[0]!=0:
#         judgements_eq=torch.Tensor(eq_mat).cuda()
#         judgements_ineq=torch.Tensor(ineq_mat).cuda()
#         # print("judgements_eq",judgements_eq.shape)
#         # print("judgements_ineq",judgements_ineq.shape)
#         # judgements_eq torch.Size([41, 6])
#         # judgements_ineq torch.Size([75, 6])

#         # ================================================================================
#         R_vec=refl_img.view(-1).unsqueeze(1)
#         # print("R_vec",R_vec.shape)
#         # R_vec torch.Size([174592, 1])

#         # ================================================================================
#         y_1=torch.floor(judgements_eq[:,1]).long()
#         y_2=torch.floor(judgements_eq[:,3]).long()
#         # print("y_1",y_1)
#         # print("y_2",y_2)
#         # y_1 tensor([276, 305, 276, 241, 328, 296, 286, 305, 242, 296, 233, 252, 241, 326,
#         #     330, 330, 262, 289, 208, 177, 209, 208], device='cuda:0')
#         # y_2 tensor([241, 296, 326, 173, 276, 330, 330, 250, 183, 282, 282, 305, 208, 262,
#         #         314, 330, 300, 183, 211, 211, 164, 247], device='cuda:0')

#         x_1=torch.floor(judgements_eq[:,0]).long()
#         x_2=torch.floor(judgements_eq[:,2]).long()

#         point_1_idx_linaer=y_1*cols+x_1
#         point_2_idx_linear=y_2*cols+x_2
#         # print("point_1_idx_linaer",point_1_idx_linaer)
#         # print("point_2_idx_linear",point_2_idx_linear)
#         # point_1_idx_linaer tensor([ 60288,  89669, 132239,  70056, 157086, 139114, 132208,  21582,   6161,
#         #       6821,   7062,  31953,  85647, 110930,  10412,  44277,   9775,  90690,
#         #      31580,  95616, 110712, 118098,  95655,  99783, 122270, 116126, 161308,
#         #      14647,  60275,  60290, 139129, 161342,  29607,  30119,  70055],
#         #    device='cuda:0')
#         # point_2_idx_linear tensor([ 82855,  90744, 110767,  95716, 139208, 137118, 127151,   9858,   8357,
#         #         6598,   9670,  10486, 110750,  82822,  28369,  62775,  27214,  60997,
#         #         14722,  82856,  89743,  82771,  60328, 116203, 139238, 139207, 139397,
#         #         44380,  61351,  31655, 156574, 133765,  60354,  60398,  60388],
#         #     device='cuda:0')

#         # Extract all pairs of comparisions
#         # points_1_vec=torch.index_select(R_vec,0,Variable(point_1_idx_linaer,requires_grad=False))
#         # points_2_vec=torch.index_select(R_vec,0,Variable(point_2_idx_linear,requires_grad=False))
#         points_1_vec=torch.index_select(R_vec,0,point_1_idx_linaer)
#         points_2_vec=torch.index_select(R_vec,0,point_2_idx_linear)
#         # print("points_1_vec",points_1_vec.shape)
#         # print("points_2_vec",points_2_vec.shape)
#         # points_1_vec torch.Size([10, 325])
#         # points_2_vec torch.Size([10, 325])

#         # weight=Variable(judgements_eq[:,5],requires_grad=False)
#         weight=judgements_eq[:,5]
#         # print("weight",weight)
#         # weight torch.Size([35])

#         # Compute loss
#         sq=torch.pow(points_1_vec-points_2_vec,2)
#         # print("sq",sq.shape)
#         # sq torch.Size([10, 35])
#         m=torch.mean(sq,0)
#         # print("m",m.shape)
#         # m torch.Size([35])
#         eq_loss=torch.sum(torch.mul(weight,m))
#         # print("eq_loss",eq_loss)
#         num_valid_eq=judgements_eq.shape[0]
#         # print("num_valid_eq",num_valid_eq)

#         # ================================================================================
#         y_1=torch.floor(judgements_ineq[:,1]).long()
#         y_2=torch.floor(judgements_ineq[:,3]).long()

#         x_1=torch.floor(judgements_ineq[:,0]).long()
#         x_2=torch.floor(judgements_ineq[:,2]).long()

#         point_1_idx_linaer=y_1*cols+x_1
#         point_2_idx_linear=y_2*cols+x_2

#         # points_1_vec=torch.index_select(R_vec,0,Variable(point_1_idx_linaer,requires_grad=False))
#         # points_2_vec=torch.index_select(R_vec,0,Variable(point_2_idx_linear,requires_grad=False))
#         points_1_vec=torch.index_select(R_vec,0,point_1_idx_linaer)
#         points_2_vec=torch.index_select(R_vec,0,point_2_idx_linear)
#         # weight=Variable(judgements_ineq[:,5],requires_grad=False)
#         weight=judgements_ineq[:,5]

#         # relu_layer=nn.ReLU(True)
#         # clamp=torch.clamp(points_2_vec-points_1_vec+tau,0.0)
#         ineq_loss=torch.sum(torch.mul(weight,torch.pow(points_2_vec-points_1_vec+tau,2)))

#         # num_included=torch.sum(torch.ge(points_2_vec.data-points_1_vec.data,-tau).float().cuda())
#         # ge=torch.ge(points_2_vec.data-points_1_vec.data,-tau).float()
#         # ge=torch.ge(points_2_vec-points_1_vec,-tau).float()
#         # num_included=torch.sum(ge)
#         # num_valid_ineq+=num_included
#         num_valid_ineq=judgements_ineq.shape[0]
#         final_lo=(eq_loss/(num_valid_eq+1e-8))+(ineq_loss/(num_valid_ineq+1e-8))
#         # final_lo tensor(12.2622, device='cuda:0', grad_fn=<ThAddBackward>)
#         # print("final_lo",final_lo)
#         return final_lo
#     else:
#         final_lo=torch.Tensor([0.0]).squeeze().cuda()
#         # print("final_lo",final_lo)
#         return final_lo
# endregion original iiw loss, updated with log, update one image at once


# region original, calculate loss per pixel
def SVM_hinge_loss(
  o_p_ref,o_iiw_ori_img,hrjf,one_p_iiw_ds,applied_DA,one_iiw_img_after_DA_before_rs):
  """
  Act
    Mu function based on Eq 2 from supplementary material
    which is used with confidence to calculate domain filter loss L_df
  """

  # c comparisons: JSON comparison file for 1 GT image
  comparisons=hrjf.judgements['intrinsic_comparisons']

  # c o_whdr: whdr score from one image
  o_whdr=eval_single_comparison_with_SVM_hinge_loss(
      o_p_ref,o_iiw_ori_img,comparisons,hrjf,one_p_iiw_ds,applied_DA,one_iiw_img_after_DA_before_rs)

  # ================================================================================
  return o_whdr
# endregion original, calculate loss per pixel

# region original, calculate loss per image
# def SVM_hinge_loss(one_iiw_pred,one_iiw_gt):
#     """
#     Act
#       Mu function based on Eq 2 from supplementary material
#       which is used with confidence to calculate domain filter loss L_df
#     """
#     # c o_whdr: whdr score from one image
#     o_whdr=eval_single_comparison_with_SVM_hinge_loss(one_iiw_pred,one_iiw_gt)
#     return o_whdr
# endregion original, calculate loss per image

# temp=0.0
# for i in all_pixel:
#     l1loss=l1loss(ref[i]-ref_sub_mat)
#     v_weight=1/2([spatial_position,intensity,chrom1,chrom2]-sub_mat).T * cov_mat.T * ([spatial_position,intensity,chrom1,chrom2]-sub_mat)
#     temp+=v_weight*l1loss
# rsmooth=temp/num_pix

# region Loss functions from CGINTRINSIC code
def Data_Loss(log_prediction,mask,log_gt):
    # c N: summed values except for 0 values in mask
    N=torch.sum(mask)
    # c log_diff: calculated log difference
    log_diff=log_prediction-log_gt
    # c log_diff: make 0s in log difference by using mask
    log_diff=torch.mul(log_diff,mask)

    s1=torch.sum(torch.pow(log_diff,2))/N
    s2=torch.pow(torch.sum(log_diff),2)/(N*N)  
    data_loss=s1-s2
    return data_loss

def L2GradientMatchingLoss(log_prediction,mask,log_gt):
    # c N: summed values except for 0 values in mask
    N=torch.sum(mask)
    # c log_diff: calculated log difference
    log_diff=log_prediction-log_gt
    # c log_diff: make 0s in log difference by using mask
    log_diff=torch.mul(log_diff,mask)
    
    # c v_gradient: vertical gradient image
    v_gradient=torch.pow(log_diff[:,:,0:-2,:]-log_diff[:,:,2:,:],2)
    # Make 0 in mask
    v_mask=torch.mul(mask[:,:,0:-2,:],mask[:,:,2:,:])
    # Make 0 in vertical gradient image
    v_gradient=torch.mul(v_gradient,v_mask)

    # c h_gradient: horizontal gradient image
    h_gradient=torch.pow(log_diff[:,:,:,0:-2]-log_diff[:,:,:,2:],2)
    # Make 0 in mask
    h_mask=torch.mul(mask[:,:,:,0:-2],mask[:,:,:,2:])
    # Make 0 in vertical gradient image
    h_gradient=torch.mul(h_gradient,h_mask)

    gradient_loss=(torch.sum(h_gradient)+torch.sum(v_gradient))
    gradient_loss=gradient_loss/N

    return gradient_loss

def HuberLoss(prediction,gt):
    tau=1.0

    # c num_valid: sum valid pixel values in mask
    # num_valid=torch.sum(mask)
    num_valid=prediction.shape[0]*prediction.shape[2]*prediction.shape[3]

    # c diff_L1: abs of difference
    diff_L1=torch.abs(prediction-gt)
    # print("diff_L1",diff_L1)
    # diff_L1=diff_L1.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,:]
    # plt.imshow(diff_L1)
    # plt.show()

    # c diff_L2: pow of difference
    diff_L2=torch.pow(prediction-gt,2)
    # print("diff_L2",diff_L2)
    # diff_L2=diff_L2.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,:]
    # plt.imshow(diff_L2)
    # plt.show()

    # Make 0 in areas where diff_L1<=tau, otherwise 1
    mask_L2=torch.le(diff_L1,tau).float()
    # mask_L2=mask_L2.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,:]
    # plt.imshow(mask_L2)
    # plt.show()
    
    mask_L1=1.0-mask_L2
    # mask_L1=mask_L1.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,:]
    # plt.imshow(mask_L1)
    # plt.show()

    # cm_d_L2: masked diff_L2
    m_d_L2=torch.mul(mask_L2,diff_L2)
    # m_d_L2=m_d_L2.detach().cpu().numpy().transpose(0,2,3,1)[0,:,:,:]
    # plt.imshow(m_d_L2)
    # plt.show()

    # cm_m_d_L2:maskedm_d_L2
    # m_m_d_L2=torch.mul(mask,m_d_L2)
    L2_loss=0.5*torch.sum(m_d_L2)
    # print("L2_loss",L2_loss)
    # L2_loss tensor(929.)
  
    # cm_d_L1: masked diff_L1
    m_d_L1=torch.mul(mask_L1,diff_L1)
    # c m_m_d_L1: masked m_d_L1
    # m_m_d_L1=torch.mul(mask,m_d_L1)
    L1_loss=torch.sum(m_d_L1)-0.5
    # print("L1_loss",L1_loss)
    # L1_loss tensor(36267184.)

    final_loss=(L2_loss+L1_loss)/num_valid
    # print("final_loss",final_loss)
    # final_loss tensor(207.7307)

    return final_loss

def LocalAlebdoSmoothenessLoss(R,targets):
    R=R.cuda()
    targets=targets.cuda()
    """
    import numpy as np
    x=np.arange(-1,2)
    y=np.arange(-1,2)
    X,Y=np.meshgrid(x,y)
    array([[-1, 0, 1],
           [-1, 0, 1],
           [-1, 0, 1]])
    array([[-1,-1,-1],
           [ 0, 0, 0],
           [ 1, 1, 1]])
    """
    # ch: height of R
    h=R.size(2)
    # cw: width of R
    w=R.size(3)
    # c num_c: channel of R
    num_c=R.size(1)

    h_w_s=1
    total_loss=torch.cuda.FloatTensor(1)
    total_loss[0]=0
    # [:,:,1:h-1,1:w-1]
    R_center=R[:,
                :,
                h_w_s+Y[h_w_s,h_w_s]:h-h_w_s+Y[h_w_s,h_w_s],
                h_w_s+X[h_w_s,h_w_s]:w-h_w_s+X[h_w_s,h_w_s]]

    c_idx=0
    # (0,3)
    for k in range(0,h_w_s*2+1):
        # (0,3)
        for l in range(0,h_w_s*2+1):
            # albedo_weights=targets["r_w_s"+str(scale_idx)][:,c_idx,:,:].unsqueeze(1).repeat(1,num_c,1,1).float().cuda()
            albedo_weights=torch.Tensor([1.5]).cuda()
            # [:,:,1+Y[0,0]:h-1+Y[0,0],1+X[0,0]:h-1+X[0,0]]
            R_N=R[:,
                    :,
                    h_w_s+Y[k,l]:h-h_w_s+Y[k,l],
                    h_w_s+X[k,l]:w-h_w_s+X[k,l]]
            # mask_N=M[:,:,h_w_s+self.Y[k,l]:h-h_w_s+self.Y[k,l],h_w_s+self.X[k,l]:w-h_w_s+self.X[k,l]]
            # composed_M=torch.mul(mask_N,mask_center)
            # albedo_weights=torch.mul(albedo_weights,composed_M)
            r_diff=torch.mul(albedo_weights,torch.abs(R_center-R_N))
            
            total_loss=total_loss+torch.mean(r_diff)
            c_idx=c_idx+1
    lo_final=total_loss/(8.0*num_c)
    # print("lo_final",lo_final)
    # lo_final tensor([4.1298], device='cuda:0')

    return lo_final

def MaskLocalSmoothenessLoss(R,M,targets):
  dense_R_gt_img_tc=targets
  dense_pred_R_imgs_direct=R
  cgmit_mask_3c_imgs_tc=M
  
  h=R.size(2)
  w=R.size(3)
  num_c=R.size(1)

  h_w_s=1
  total_loss=Variable(torch.cuda.FloatTensor(1))
  total_loss[0]=0

  mask_center=M[:,
                :,
                h_w_s+self.Y[h_w_s,h_w_s]:h-h_w_s+self.Y[h_w_s,h_w_s],
                h_w_s+self.X[h_w_s,h_w_s]:w-h_w_s+self.X[h_w_s,h_w_s]]

  R_center=R[:,
              :,
              h_w_s+self.Y[h_w_s,h_w_s]:h-h_w_s+self.Y[h_w_s,h_w_s],
              h_w_s+self.X[h_w_s,h_w_s]:w-h_w_s+self.X[h_w_s,h_w_s]]

  c_idx=0

  for k in range(0,h_w_s*2+1):
      for l in range(0,h_w_s*2+1):
          # albedo_weights=Variable(targets["r_w_s"+str(scale_idx)][:,c_idx,:,:].unsqueeze(1).repeat(1,num_c,1,1).float().cuda(),requires_grad=False)
          R_N=R[:,
                  :,
                  h_w_s+self.Y[k,l]:h-h_w_s+self.Y[k,l],
                  h_w_s+self.X[k,l]:w-h_w_s+self.X[k,l]]
          mask_N=M[:,
                      :,
                      h_w_s+self.Y[k,l]:h-h_w_s+self.Y[k,l],
                      h_w_s+self.X[k,l]:w-h_w_s+self.X[k,l]]

          composed_M=torch.mul(mask_N,mask_center)

          # albedo_weights=torch.mul(albedo_weights,composed_M)

          r_diff=torch.mul(composed_M,torch.pow(R_center-R_N,2))
          total_loss=total_loss+torch.mean(r_diff)
          c_idx=c_idx+1

  return total_loss/(8.0*num_c)

def SmoothLoss(prediction_n):
  # N=torch.sum(mask[:,0,:,:])
  # N=prediction_n.shape[0]*prediction_n.shape[2]*prediction_n.shape[3]
  N=torch.sum(prediction_n[:,0,:,:])

  # horizontal angle difference
  # h_mask=torch.mul(mask[:,:,:,0:-2],mask[:,:,:,2:])
  h_gradient=torch.sum(torch.mul(prediction_n[:,:,:,0:-2],prediction_n[:,:,:,2:]),1)
  # print("h_gradient",h_gradient.shape)
  # h_gradient torch.Size([1, 341, 510])
  h_gradient_loss=1-torch.sum(h_gradient)/N
  # print("h_gradient_loss",h_gradient_loss)
  # h_gradient_loss tensor(-82494.7891)

  # Vertical angle difference
  # v_mask=torch.mul(mask[:,:,0:-2,:],mask[:,:,2:,:])
  mul_term=torch.mul(prediction_n[:,:,0:-2,:],prediction_n[:,:,2:,:])
  # print("mul_term",mul_term.shape)
  v_gradient=torch.sum(mul_term,1)
  # print("v_gradient",v_gradient)
  v_gradient_loss=1-torch.sum(v_gradient)/N
  # print("v_gradient_loss",v_gradient_loss)
  # v_gradient_loss tensor(-82278.)

  gradient_loss=h_gradient_loss+v_gradient_loss
  # print("gradient_loss",gradient_loss)
  # gradient_loss tensor(-164772.7812)
  # gradient_loss tensor(-915.3240)

  return gradient_loss

def GradientLoss(prediction_n,gt_n):
  # softsign=torch.nn.Softsign()
  # prediction_n=softsign(prediction_n)
  # gt_n=softsign(gt_n)

  # c N: summed values except for 0 values in mask
  # N=torch.sum(mask)
  N=prediction_n.shape[0]*prediction_n.shape[2]*prediction_n.shape[3]

  h_gradient_pred=torch.abs(prediction_n[:,:,:,:-2]-prediction_n[:,:,:,2:])
  # for i in range(h_gradient.shape[0]):
  #     h_gradient_t=h_gradient[i,:,:,:].detach().cpu().numpy().squeeze()
  #     plt.imshow(h_gradient_t,cmap="gray")
  #     plt.show()
  h_gradient_gt=torch.abs(gt_n[:,:,:,:-2]-gt_n[:,:,:,2:])

  import scipy.misc
  for i in range(h_gradient_pred.shape[0]):
    scipy.misc.imsave('./img_out/h_gradient_pred_'+str(i)+'.png',h_gradient_pred[i,:,:,:].detach().cpu().numpy().squeeze())
  for i in range(h_gradient_gt.shape[0]):
    scipy.misc.imsave('./img_out/h_gradient_gt_'+str(i)+'.png',h_gradient_gt[i,:,:,:].detach().cpu().numpy().squeeze())

  h_gradient_diff=torch.sum(torch.abs(h_gradient_pred-h_gradient_gt))

  # ================================================================================
  v_gradient_pred=torch.abs(prediction_n[:,:,:-2,:]-prediction_n[:,:,2:,:])
  v_gradient_gt=torch.abs(gt_n[:,:,:-2,:]-gt_n[:,:,2:,:])
  v_gradient_diff=torch.sum(torch.abs(v_gradient_pred-v_gradient_gt))

  for i in range(v_gradient_pred.shape[0]):
    scipy.misc.imsave('./img_out/v_gradient_pred_'+str(i)+'.png',v_gradient_pred[i,:,:,:].detach().cpu().numpy().squeeze())
  for i in range(v_gradient_gt.shape[0]):
    scipy.misc.imsave('./img_out/v_gradient_gt_'+str(i)+'.png',v_gradient_gt[i,:,:,:].detach().cpu().numpy().squeeze())

  gradient_diff=(h_gradient_diff+v_gradient_diff)/(N*2.0)
  # print("gradient_loss",gradient_loss)
  # gradient_loss tensor(0.0348, device='cuda:0', grad_fn=<DivBackward0>)

  return gradient_diff

def GradientLoss_using_F_conv2d_sobel(prediction_n,gt_n,mask_imgs_dense):
  # c N: summed values except for 0 values in mask
  N=torch.sum(mask_imgs_dense)
  # print("N",N)

  # N=prediction_n.shape[0]*prediction_n.shape[2]*prediction_n.shape[3]

  h_gradient_pred,v_gradient_pred=utils_image.get_gradient_of_image(prediction_n)

  # print(h_gradient_pred.shape)
  # print(v_gradient_pred.shape)
  # torch.Size([10, 1, 256, 256])
  # torch.Size([10, 1, 256, 256])
  # import scipy.misc
  # for i in range(h_gradient_pred.shape[0]):
  #   scipy.misc.imsave('./png_out/h_gradient_pred_'+str(i)+'.png',h_gradient_pred[i,:,:,:].detach().cpu().numpy().squeeze())
  # for i in range(v_gradient_pred.shape[0]):
  #   scipy.misc.imsave('./png_out/v_gradient_pred_'+str(i)+'.png',v_gradient_pred[i,:,:,:].detach().cpu().numpy().squeeze())

  h_gradient_gt,v_gradient_gt=utils_image.get_gradient_of_image(gt_n)

  # for i in range(h_gradient_gt.shape[0]):
  #   scipy.misc.imsave('./png_out/h_gradient_gt_'+str(i)+'.png',h_gradient_gt[i,:,:,:].detach().cpu().numpy().squeeze())
  # for i in range(v_gradient_gt.shape[0]):
  #   scipy.misc.imsave('./png_out/v_gradient_gt_'+str(i)+'.png',v_gradient_gt[i,:,:,:].detach().cpu().numpy().squeeze())

  # ================================================================================
  # @ Perform mask    
  h_gradient_gt=torch.mul(h_gradient_pred,mask_imgs_dense)
  h_gradient_gt=torch.mul(h_gradient_gt,mask_imgs_dense)
  v_gradient_pred=torch.mul(v_gradient_pred,mask_imgs_dense)
  v_gradient_gt=torch.mul(v_gradient_gt,mask_imgs_dense)

  # ================================================================================
  h_gradient_diff=torch.sum(torch.abs(h_gradient_pred-h_gradient_gt))
  v_gradient_diff=torch.sum(torch.abs(v_gradient_pred-v_gradient_gt))
  
  # ================================================================================
  gradient_diff=(h_gradient_diff+v_gradient_diff)/(N*2.0)
  # print("gradient_loss",gradient_loss)
  # gradient_loss tensor(0.0348, device='cuda:0', grad_fn=<DivBackward0>)

  return gradient_diff

def GradientLoss_3c(prediction_n,gt_n):
  # c N: summed values except for 0 values in mask
  # N=torch.sum(mask)
  N=prediction_n.shape[0]*prediction_n.shape[2]*prediction_n.shape[3]

  h_gradient_pred=torch.abs(prediction_n[:,:,:,:-1]-prediction_n[:,:,:,1:])
  # print("h_gradient_pred",h_gradient_pred.shape)
  # torch.Size([10, 3, 256, 254])
  h_gradient_gt=torch.abs(gt_n[:,:,:,:-1]-gt_n[:,:,:,1:])
  # print("h_gradient_gt",h_gradient_gt.shape)
  # torch.Size([10, 3, 256, 254])
  h_gradient_diff=torch.sum(torch.abs(h_gradient_pred-h_gradient_gt))

  # for i in range(h_gradient_pred.shape[0]):
  #   scipy.misc.imsave('./png_out/h_gradient_pred3c_'+str(i)+'.png',h_gradient_pred[i,:,:,:].detach().cpu().numpy().squeeze().transpose(1,2,0))
  # for i in range(h_gradient_gt.shape[0]):
  #   scipy.misc.imsave('./png_out/h_gradient_gt3c_'+str(i)+'.png',h_gradient_gt[i,:,:,:].detach().cpu().numpy().squeeze().transpose(1,2,0))
  
  # ================================================================================
  v_gradient_pred=torch.abs(prediction_n[:,:,:-1,:]-prediction_n[:,:,1:,:])
  v_gradient_gt=torch.abs(gt_n[:,:,:-1,:]-gt_n[:,:,1:,:])
  v_gradient_diff=torch.sum(torch.abs(v_gradient_pred-v_gradient_gt))

  # for i in range(v_gradient_pred.shape[0]):
  #   scipy.misc.imsave('./png_out/v_gradient_pred3c_'+str(i)+'.png',v_gradient_pred[i,:,:,:].detach().cpu().numpy().squeeze().transpose(1,2,0))
  # for i in range(v_gradient_gt.shape[0]):
  #   scipy.misc.imsave('./png_out/v_gradient_gt3c_'+str(i)+'.png',v_gradient_gt[i,:,:,:].detach().cpu().numpy().squeeze().transpose(1,2,0))

  gradient_diff=(h_gradient_diff+v_gradient_diff)/(N*2.0)
  # print("gradient_diff",gradient_diff)

  return gradient_diff

def GradientLoss_3c_using_F_conv2d_sobel(prediction_n,gt_n,mask_imgs_dense):
  # c N: summed values except for 0 values in mask
  N=torch.sum(mask_imgs_dense)
  # N=prediction_n.shape[0]*prediction_n.shape[2]*prediction_n.shape[3]

  # ================================================================================
  h_gradient_pred,v_gradient_pred=utils_image.get_gradient_of_image(prediction_n)
  h_gradient_gt,v_gradient_gt=utils_image.get_gradient_of_image(gt_n)
  # for i in range(h_gradient.shape[0]):
  #     h_gradient_t=h_gradient[i,:,:,:].detach().cpu().numpy().squeeze()
  #     plt.imshow(h_gradient_t,cmap="gray")
  #     plt.show()

  # ================================================================================
  # @ Perform mask    
  h_gradient_pred=torch.mul(h_gradient_pred,mask_imgs_dense)
  h_gradient_gt=torch.mul(h_gradient_gt,mask_imgs_dense)

  v_gradient_pred=torch.mul(v_gradient_pred,mask_imgs_dense)
  v_gradient_gt=torch.mul(v_gradient_gt,mask_imgs_dense)

  # ================================================================================
  h_gradient_diff=torch.sum(torch.abs(h_gradient_pred-h_gradient_gt))
  v_gradient_diff=torch.sum(torch.abs(v_gradient_pred-v_gradient_gt))

  # ================================================================================
  gradient_diff=(h_gradient_diff+v_gradient_diff)/N*2.0
  # print("gradient_diff",gradient_diff)
  # tensor(965598.2500, device='cuda:0', grad_fn=<AddBackward0>)

  return gradient_diff

def L2Loss_CG(prediction_n,gt):
  # num_valid=torch.sum(mask)
  num_valid=float(prediction_n.shape[0]*prediction_n.shape[2]*prediction_n.shape[3])

  diff=torch.pow(prediction_n-gt,2)
  # diff=torch.where(torch.isnan(diff),torch.Tensor([0.0001]).cuda(),diff)
  diff=torch.sum(diff)/num_valid

  # print("diff",diff)
  return diff

# end
# endregion Loss functions from CGINTRINSIC code

