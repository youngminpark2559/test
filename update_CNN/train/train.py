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

# ---------------
if torch.cuda.is_available():
    device="cuda:0"
else:
    device="cpu"

# device=torch.device(device)
device=torch.device("cuda:0")
checkpoint_path="./direct_intrinsic_net.pth.tar"

currentdir = "/mnt/1T-5e7/mycodehtml/cv/IID/CNN_creating_R_intensity/train"
network_dir="/mnt/1T-5e7/mycodehtml/cv/IID/CNN_creating_R_intensity/networks"
sys.path.insert(0,network_dir)
loss_function_dir="/mnt/1T-5e7/mycodehtml/cv/IID/CNN_creating_R_intensity/loss_functions"
sys.path.insert(0,loss_function_dir)
utils_dir="/mnt/1T-5e7/mycodehtml/cv/IID/CNN_creating_R_intensity/utils"
sys.path.insert(0,utils_dir)

import networks as networks
import loss_functions as loss_functions
import utils as utils

def train():
    # c transformer: created transformer for input images
    transformer=transforms.Compose(
        [transforms.Resize((224,224)),
         transforms.ToTensor()])

    # c train_dataset: created train_dataset instance
    train_dataset=torchvision.datasets.ImageFolder(
        root=args.train_dir,transform=transformer)

    # c nei: number of entire loaded images
    nei=print(len(train_dataset))
    # 2 images

    # c train_dataloader: created train_dataloader
    train_dataloader=DataLoader(
        train_dataset,batch_size=int(args.batch_size),shuffle=False,num_workers=2)

    if args.continue_training==True:
        direct_intrinsic_net,checkpoint,optimizer=utils.net_generator(args)
    else:
        direct_intrinsic_net,optimizer=utils.net_generator(args)

    # Iterates as much as epoch
    for ep in range(int(args.epoch)):
        # c dataiter: dataset iterator
        dataiter=iter(train_dataloader)

        # 2 batches*1 iterations=process 2 images
        for itr in range(int(args.iteration)):
            # Initialize all gradients of Variables as zero
            optimizer.zero_grad()

            # c o_images: original images
            # c o_labels: original labels
            o_images,o_labels=dataiter.next()
            # print("o_images",type(o_images))
            # o_images <class 'torch.Tensor'>
            
            # print("o_images",o_images.shape)
            # o_images torch.Size([2, 3, 224, 224])

            # ---------------
            # Wrap image torch tensors by Variable and upload image Variables onto GPU
            o_images=Variable(o_images).to(device)

            # ---------------
            # Forward pass

            # c prir: predicted reflectance gray scale image containing intensity scalar value r
            prir=direct_intrinsic_net(o_images)
            
            # print("prir",prir.grad)

            # ---------------
            # Calculate loss

            json_list=glob.glob("/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data3/train/*.json")
            json_list=natsort.natsorted(json_list,reverse=False)
            # print("json_list",json_list)
            # ['/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data3/63.json', 
            #  '/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data3/54.json']

            png_list=glob.glob("/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data3/train/*.png")
            png_list=natsort.natsorted(png_list,reverse=False)

            # c en_i_loss: sum of all whdrs from all images, entire images loss
            en_i_loss=0

            # torch.Size([2, 1, 224, 224])
            # c o_batch: one batch
            for o_batch in range(prir.shape[0]):

                img_path=png_list[o_batch]
                img = Image.open(img_path)
                ori_img_s = np.array(img).shape
                # print("ori_img_s",ori_img_s)
                # (341, 512, 3)

                # # c o_p_ref: one predicted reflectance image
                # o_p_ref=prir[o_batch,:,:,:].detach().cpu().numpy()

                # o_p_ref = cv2.normalize(o_p_ref, None, alpha=0.01, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                # o_p_ref=Variable(torch.Tensor(o_p_ref))

                # o_prir=prir[o_batch,:,:,:].detach().cpu().numpy()
                # o_prir=resize(o_prir,(1,ori_img_s[0],ori_img_s[1]), order=1, preserve_range=True)

                # o_p_ref=Variable(torch.Tensor(o_prir))

                o_p_ref=prir[o_batch,:,:,:]


                # c o_gt: one GT json file for above predicted reflectance image
                o_gt=json_list[o_batch]
                # print("o_gt",o_gt)
                # /mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data3/train/54.json
                # /mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/data3/train/63.json

                hrjf=loss_functions.HumanReflectanceJudgements.from_file(o_gt)

                # c o_i_loss: whdr of one image
                # my_loss_f=loss_functions.Direct_Intrinsic_Net_Loss()
                # o_i_loss=my_loss_f(o_p_ref,hrjf,ori_img_s)

                # o_i_loss=loss_functions.Direct_Intrinsic_Net_Loss_F.forward(o_p_ref,hrjf,ori_img_s)
                o_i_loss=loss_functions.SVM_hinge_loss(o_p_ref,hrjf,ori_img_s)
                
                # print("o_i_loss",o_i_loss)
                print("o_i_loss",type(o_i_loss))

                en_i_loss+=o_i_loss

            # print("en_i_loss p",type(en_i_loss)) # <class 'torch.Tensor'>

            # en_i_loss=Variable(en_i_loss)

            # print("en_i_loss",en_i_loss)

            # print("list(direct_intrinsic_net.parameters())",list(direct_intrinsic_net.parameters()))

            # ---------------
            # Update network based on loss

            print("en_i_loss",en_i_loss)

            # Backpropagation based on loss
            en_i_loss.backward()

            print("en_i_loss.grad",en_i_loss.grad)

            # for param in direct_intrinsic_net.parameters():
            #     print("param",param)
            #     print("param.grad",param.grad)
            #     print("param.requires_grad",param.requires_grad) # True
            #     print(param.grad.data.sum())
                # AttributeError: 'NoneType' object has no attribute 'data'

            # for p in direct_intrinsic_net.parameters():
            #     if p.grad is not None:
            #         print("p.grad.data",p.grad.data)
            #         print("p.grad.data.sum()",p.grad.data.sum())

            # start debugger
            # import pdb; pdb.set_trace()

            # Update network based on backpropagation
            optimizer.step()

            # Save checkpoint if is a new best
            utils.save_checkpoint(
                {'epoch': ep + 1,
                 'state_dict': direct_intrinsic_net.state_dict(),
                 'optimizer' : optimizer.state_dict(),}, 
                checkpoint_path)
            print("Saved model at end of iteration")
            
            # ---------------
            # Visualize

            # print("prir",prir.shape) # torch.Size([2, 1, 224, 224])

            # Iterates as much as batch size
            for o_batch in range(prir.shape[0]):
                # c pred_inten: one predicted gray scale reflectance image
                pred_inten=prir[o_batch,:,:,:].detach().cpu().numpy().squeeze()
                # print("pred_inten",pred_inten.shape)
                # pred_inten (224, 224)

                if ep==400:
                    plt.imshow(pred_inten,cmap="gray")
                    plt.title("predicted intensity")
                    plt.show()

                ori_img=o_images[o_batch,:,:,:].detach().cpu().numpy().transpose(1,2,0)

                pred_inten = cv2.normalize(pred_inten, None, alpha=0.01, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                ori_img = cv2.normalize(ori_img, None, alpha=0.01, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                # c ref: colorized gray scale reflectance intensity to rgb image obtained by using shading and original image
                # c sha: shading image obtained by using original image and reflectance intensity
                ref,sha=utils.colorize(pred_inten,ori_img)
                # print("ref",ref.shape)
                # ref (224, 224, 3)

                # print("sha",sha.shape)
                # sha (224, 224)
                
                ref = cv2.normalize(ref, None, alpha=0.01, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                sha = cv2.normalize(sha, None, alpha=0.01, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                if ep==400:
                    plt.imshow(ref)
                    # plt.title("ref")
                    plt.show()

                if ep==400:
                    plt.imshow(sha,cmap="gray")
                    # plt.title("sha")
                    plt.show()


    # Save parameters' values at every epoch
    utils.save_checkpoint(
        {'epoch': ep + 1,
            'state_dict': direct_intrinsic_net.state_dict(),
            'optimizer' : optimizer.state_dict(),}, 
        checkpoint_path)
    print("Saved model at end of epoch")
    print("Train finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This parses arguments and then runs appropriate mode based on arguments")
    parser.add_argument(
        "--train_dir",
        help="--train_dir /home/os_user_name/test_input_dir/train/")
    parser.add_argument(
        "--iteration",
        default=4,
        help="number of iteration in one epoch, batch_size*iteration=one epoch, --iteration 4")
    parser.add_argument(
        "--epoch",
        default=2,
        help="number of epoch, --epoch 2")
    parser.add_argument(
        "--batch_size",
        default=2,
        help="--batch_size 2")
    parser.add_argument(
        "--continue_training",
        default=False,
        help="True or False")
    parser.add_argument(
        "--trained_network_path")
    parser.add_argument(
        "--use_pretrained_resnet152",
        default=False,
        help="True or False")
    # args = parser.parse_args(sys.argv[1:])
    args = parser.parse_args()

    train()
