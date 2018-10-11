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
import glob
from skimage.transform import resize

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

utils_dir="/mnt/1T-5e7/mycodehtml/cv/IID/CNN_creating_R_intensity/utils"
sys.path.insert(0,utils_dir) 
import utils

delta = 0.1  # threshold for "more or less equal"
xi = 0.0  # margin in Hinge
ratio = 1.0  # ratio of evaluated comparisons
eval_dense = 1  # evaluate dense labels?

class HumanReflectanceJudgements(object):

    def __init__(self, judgements):
        if not isinstance(judgements, dict):
            raise ValueError("Invalid judgements: %s" % judgements)

        self.judgements = judgements
        self.id_to_points = {p['id']: p for p in self.points}

    @staticmethod
    def from_file(filename):
        judgements = json.load(open(filename))
        # c hrj: human reflectance judgement json file
        hrj=HumanReflectanceJudgements(judgements)
        return hrj

    @property
    def points(self):
        return self.judgements['intrinsic_points']

    @property
    def comparisons(self):
        return self.judgements['intrinsic_comparisons']


def eval_single_comparison_with_SVM_hinge_loss(o_p_ref,comparisons,hrjf,ori_img_s):
    # print("comparisons",comparisons)

    # print("hrjf",hrjf)
    # hrjf <utils.HumanReflectanceJudgements object at 0x7f46e15dc748>

    # print("hrjf",dir(hrjf))

    # resized = scipy.misc.imresize(refl_img, (height, width))
    # prediction_R_np = resize(refl_img, (o_h,o_w), order=1, preserve_range=True)

    # print("o_p_ref",o_p_ref.shape)
    # torch.Size([1, 224, 224])

    # print("ori_img_s",ori_img_s)
    # ori_img_s (341, 512, 3)

    # c refl_img: one predicted reflectance image
    o_p_ref=o_p_ref.detach().cpu().numpy()
    refl_img=resize(o_p_ref,(1,ori_img_s[0],ori_img_s[1]), order=1, preserve_range=True)
    # print("refl_img p",refl_img.shape) # 
    # refl_img p (512, 3, 224)

    # H and W of predicted reflectance image
    rows,cols=refl_img.shape[1:3]
    # print("rows,cols",rows,cols) # 224, 224
    # rows,cols 3 224

    refl_img=Variable(torch.Tensor(refl_img))

    # c error_sum: sum all errors from all comparisons in one image
    error_sum=0
    # c weight_sum: sum all weights from all comparisons in one image
    weight_sum=0

    # JSON GT for 1 image, 
    # containing all relative reflectance comparisons information
    # c c: one comparison from 1 image's all comparisons
    for c in comparisons:
        # print("c",c)

        # print("hrjf.id_to_points",hrjf.id_to_points)
        
        # c n_po1: number of point1
        n_po1=c['point1']

        # Point1 from one comparison
        point1 = hrjf.id_to_points[n_po1]
        # print("point1",point1)

        n_po2=c['point2']

        # Point2 from one comparison
        point2 = hrjf.id_to_points[n_po2]
        # print("point2",point2)

        # Darker information from one comparison
        darker = c['darker']
        # print("darker",darker)
        # darker E
        
        # Weight information from one comparison
        weight = c['darker_score'] # 1.14812035203497
        # print("weight",weight)

        # ---------------
        # Check exception

        if not point1['opaque'] or not point2['opaque']:
            # Pass this judgement
            continue
        # weight < 0 or weight is None -> invalid darker_score so pass
        if weight < 0 or weight is None:
            raise ValueError("Invalid darker_score: %s" % weight)
        if darker not in ('1', '2', 'E'):
            raise ValueError("Invalid darker: %s" % darker)
        
        # ---------------
        # x,y coordinates from point1 and point2
        # darker information (which point is darkerer (brighter, or same) reflectance) 
        # from one comparison
        x1,y1,x2,y2,darker=int(point1['x']*cols),int(point1['y']*rows),int(point2['x']*cols),int(point2['y']*rows),darker
        # print("x1, y1, x2, y2, darker p",x1, y1, x2, y2, darker)

        # ---------------
        # print("refl_img p",refl_img.shape) # (3, 341, 512)

        # print("x1,y1",x1,y1)
        
        # c R1: reflectance RGB value of point1 from predicted reflectance
        # R1 = refl_img[:, x1, y1]
        R1 = refl_img[:, y1, x1]
        # print("R1",R1) # [0.3621]
        
        R2 = refl_img[:, y2, x2]
        # print("R2",R2) # [0.2999]

        b_R_k1=Variable(torch.mean(R1))
        # print("b_R_k1",b_R_k1) # 0.3621

        b_R_k2=Variable(torch.mean(R2))
        # print("b_R_k2",b_R_k2) # 0.2999

        # ---------------
        if b_R_k1==0 or b_R_k2==0:
            div_k1_k2=Variable(torch.Tensor([0.01]))
        else:
            div_k1_k2=b_R_k1/b_R_k2
        dx_inv=1/1+delta+xi
        dx=1+delta+xi
        dx_m_inv=1/1+delta-xi
        dx_m=1+delta-xi

        # c darker: j_k
        if darker=='1':
            # c ersp: error of single pair
            ersp=torch.max(Variable(torch.Tensor([0])),div_k1_k2-dx_inv)
            # print("ersp",ersp)
        elif darker=='2':
            ersp=torch.max(Variable(torch.Tensor([0])),dx-div_k1_k2)
            # print("ersp",ersp)
        elif darker=='E':
            if xi <= delta:
                ersp=torch.max(Variable(torch.Tensor([0])),dx_m_inv-div_k1_k2)
                # print("ersp",ersp)
            else:
                ersp=torch.max(Variable(torch.Tensor([0])),div_k1_k2-dx_m)
                # print("ersp",ersp)
        
        # Sum all errors from all comparisons in one image
        error_sum+=ersp

        # Sum all weights from all comparisons in one image
        weight_sum+=weight

    # print("weight_sum",weight_sum) # 103.48992721303007
    # print("error_sum",error_sum) # 103.48992721303007

    # Now, you have processed all comparisons in one image
    # If weight_sum exist
    if weight_sum:
        # c whdr: calculated whdr of one image
        whdr=error_sum/weight_sum

    # If weight_sum=0, it means there's no comparisons
    # In that case, assign 0 into whdr
    else:
        whdr = 0.0

    print("whdr",whdr)

    # Return whdr score of one image
    return whdr

def SVM_hinge_loss(o_p_ref,hrjf,ori_img_s):
    """
    Act
      Mu function based on Eq 2 from supplementary material
      which is used with confidence to calculate domain filter loss L_df
    """
    # c comparisons: JSON comparison file for 1 GT image
    comparisons=hrjf.judgements['intrinsic_comparisons']
    # print("comparisons",comparisons)

    # c o_whdr: whdr score from one image
    o_whdr=eval_single_comparison_with_SVM_hinge_loss(o_p_ref,comparisons,hrjf,ori_img_s)
    o_whdr=torch.mean(F.softplus(o_whdr))

    return o_whdr
