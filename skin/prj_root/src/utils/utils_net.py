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
import timeit
import sys,os
import glob
import natsort 
import traceback

import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

# ================================================================================
from src.networks import networks as networks

# ================================================================================
if torch.cuda.is_available():
  device="cuda:0"
else:
  device="cpu"

# device=torch.device(device)
device=torch.device("cuda:0")

# ================================================================================
# Horovod: initialize library.
# hvd.init()
# torch.manual_seed(args.seed)

# if args.use_multi_gpu=="True":
#   # Horovod: pin GPU to local rank.
#   torch.cuda.set_device(hvd.local_rank())
#   torch.cuda.manual_seed(args.seed)

# ================================================================================
gpu_ids=0
use_dropout=False
ngf=64
input_nc=3
output_nc=3
LR=0.01
WD=0.1

# ================================================================================
def get_file_list(path):
  file_list=glob.glob(path)
  file_list=natsort.natsorted(file_list,reverse=False)
  return file_list

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
  num_params=0
  for param in net.parameters():
    num_params+=param.numel()
  return num_params

def net_generator(args,num_entire_imgs):
  """
  Act
    * 
  
  Params
    * args
    You can manually create args
    by using utils_create_argument.return_argument()

    * num_entire_imgs
    Example is 8
  
  Return
    * 
  """
  lr=0.0001

  if args.use_integrated_decoders=="True":
    gen_net=networks.UNet_Deep_Shader_integrated_decoders().cuda()
    # Horovod: broadcast parameters.
    hvd.broadcast_parameters(gen_net.state_dict(),root_rank=0)

    # c use multiple GPUs?
    if args.use_multi_gpu=="True":
      num_gpu=torch.cuda.device_count()
      print("num_gpu",num_gpu)
      # DEVICE_IDS=list(range(num_gpu))
      # DEVICE_IDS=[0,1,2,3,4,5,6,7]
      # gen_encoder=nn.DataParallel(gen_encoder,device_ids=DEVICE_IDS)
      gen_net=nn.DataParallel(gen_net)
    else: # args.use_multi_gpu=="False":
      pass

    # --------------------------------------------------------------------------------
    # c configure learning rate scheduler
    if args.scheduler=="cyclic":
      optimizer=torch.optim.SGD(gen_net.parameters(),lr=0.01,momentum=0.9)
      scheduler_gen_net=utils_custom_lr_scheduler.CyclicLR(optimizer)
    elif args.scheduler=="cosine":
      optimizer=torch.optim.SGD(gen_net.parameters(),lr=0.1,momentum=0.9)
      scheduler_gen_net=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2)
    elif args.scheduler=="adamW":
      optimizer=utils_custom_lr_scheduler.AdamW(
        gen_net.parameters(),lr=LR,betas=(0.9,0.99),weight_decay=WD)
    elif args.scheduler=="adamW_cosine":
      optim_encoder=utils_adamw_plus_cosine_adamw.AdamW(
        gen_net.parameters(),lr=1e-3,weight_decay=1e-5)
      scheduler_gen_net=utils_adamw_plus_cosine_cosine.CosineLRWithRestarts(
        optimizer,int(args.batch_size),int(num_entire_imgs),restart_period=5,t_mult=1.2)
    else:
      optimizer=torch.optim.Adam(gen_net.parameters(),lr=lr)

      # Horovod: wrap optimizer with DistributedOptimizer.
      optimizer=hvd.DistributedOptimizer(
        optimizer,
        named_parameters=gen_net.named_parameters())

    # gen_net_test=networks.UNet_Deep_Shader_integrated_decoders_test().cuda()
    # optimizer_test=torch.optim.Adam(gen_net_test.parameters(),lr=lr)

    if args.use_saved_model=="True":
      checkpoint_gener_direct_rgb=torch.load(args.model_save_dir+"/gen_net.pth")
      start_epoch=checkpoint_gener_direct_rgb['epoch']
      gen_net.load_state_dict(checkpoint_gener_direct_rgb['state_dict'])
      optimizer.load_state_dict(checkpoint_gener_direct_rgb['optimizer'])
    
    gen_net_param=print_network(gen_net)
    # gen_net_param=print_network(gen_net_test)
    print("gen_net:",gen_net_param)

    return gen_net,optimizer
    # return gen_net_test,optimizer_test

  # --------------------------------------------------------------------------------
  else: # use_integrated_decoders=="False":
    # c gen_encoder: create network
    gen_encoder=networks.UNet_Deep_Shader_encoder(args)
    gen_decoder_R=networks.UNet_Deep_Shader_decoder_R(args)
    gen_decoder_S=networks.UNet_Deep_Shader_decoder_S(args)

    # c use multiple GPUs?
    if args.use_multi_gpu=="True":
      num_gpu=torch.cuda.device_count()
      print("num_gpu",num_gpu)
      # DEVICE_IDS=list(range(num_gpu))
      # DEVICE_IDS=[0,1,2,3,4,5,6,7]
      # gen_encoder=nn.DataParallel(gen_encoder,device_ids=DEVICE_IDS)
      # gen_decoder_R=nn.DataParallel(gen_decoder_R,device_ids=DEVICE_IDS)
      # gen_decoder_S=nn.DataParallel(gen_decoder_S,device_ids=DEVICE_IDS)
      gen_encoder=nn.DataParallel(gen_encoder)
      gen_decoder_R=nn.DataParallel(gen_decoder_R)
      gen_decoder_S=nn.DataParallel(gen_decoder_S)
    else: # args.use_multi_gpu=="False":
      pass
    
    # --------------------------------------------------------------------------------
    # c gen_encoder: upload network onto GPUs
    gen_encoder=gen_encoder.to(device)
    gen_decoder_R=gen_decoder_R.to(device)
    gen_decoder_S=gen_decoder_S.to(device)
    
    # --------------------------------------------------------------------------------
    # c configure learning rate scheduler
    if args.scheduler=="cyclic":
      optim_encoder=torch.optim.SGD(gen_encoder.parameters(),lr=0.01,momentum=0.9)
      optim_decoder_R=torch.optim.SGD(gen_decoder_R.parameters(),lr=0.01,momentum=0.9)
      optim_decoder_S=torch.optim.SGD(gen_decoder_S.parameters(),lr=0.01,momentum=0.9)
      scheduler_encoder=utils_custom_lr_scheduler.CyclicLR(optim_encoder)
      scheduler_decoder_R=utils_custom_lr_scheduler.CyclicLR(optim_decoder_R)
      scheduler_decoder_S=utils_custom_lr_scheduler.CyclicLR(optim_decoder_S)
    elif args.scheduler=="cosine":
      optim_encoder=torch.optim.SGD(gen_encoder.parameters(),lr=0.1,momentum=0.9)
      optim_decoder_R=torch.optim.SGD(gen_decoder_R.parameters(),lr=0.1,momentum=0.9)
      optim_decoder_S=torch.optim.SGD(gen_decoder_S.parameters(),lr=0.1,momentum=0.9)
      scheduler_encoder=torch.optim.lr_scheduler.CosineAnnealingLR(optim_encoder,T_max=2)
      scheduler_decoder_R=torch.optim.lr_scheduler.CosineAnnealingLR(optim_decoder_R,T_max=2)
      scheduler_decoder_S=torch.optim.lr_scheduler.CosineAnnealingLR(optim_decoder_S,T_max=2)
    elif args.scheduler=="adamW":
      optim_encoder=utils_custom_lr_scheduler.AdamW(
        gen_encoder.parameters(),lr=LR,betas=(0.9,0.99),weight_decay=WD)
      optim_decoder_R=utils_custom_lr_scheduler.AdamW(
        gen_decoder_R.parameters(),lr=LR,betas=(0.9,0.99),weight_decay=WD)
      optim_decoder_S=utils_custom_lr_scheduler.AdamW(
        gen_decoder_S.parameters(),lr=LR,betas=(0.9,0.99),weight_decay=WD)
    elif args.scheduler=="adamW_cosine":
      optim_encoder=utils_adamw_plus_cosine_adamw.AdamW(
        gen_encoder.parameters(),lr=1e-3,weight_decay=1e-5)
      scheduler_encoder=utils_adamw_plus_cosine_cosine.CosineLRWithRestarts(
        optim_encoder,int(args.batch_size),int(num_entire_imgs),restart_period=5,t_mult=1.2)
      optim_decoder_R=utils_adamw_plus_cosine_adamw.AdamW(
        gen_decoder_R.parameters(),lr=1e-3,weight_decay=1e-5)
      scheduler_decoder_R=utils_adamw_plus_cosine_cosine.CosineLRWithRestarts(
        optim_decoder_R,int(args.batch_size),int(num_entire_imgs),restart_period=5,t_mult=1.2)
      optim_decoder_S=utils_adamw_plus_cosine_adamw.AdamW(
        gen_decoder_S.parameters(),lr=1e-3,weight_decay=1e-5)
      scheduler_decoder_S=utils_adamw_plus_cosine_cosine.CosineLRWithRestarts(
        optim_decoder_S,int(args.batch_size),int(num_entire_imgs),restart_period=5,t_mult=1.2)
    else:
        optim_encoder=torch.optim.Adam(gen_encoder.parameters(),lr=lr)
        optim_decoder_R=torch.optim.Adam(gen_decoder_R.parameters(),lr=lr)
        optim_decoder_S=torch.optim.Adam(gen_decoder_S.parameters(),lr=lr)

    # --------------------------------------------------------------------------------
    # c load save model
    if args.use_saved_model=="True":
      checkpoint_encoder=torch.load(args.model_save_dir+"/encoder.pth")
      start_epoch=checkpoint_encoder['epoch']
      gen_encoder.load_state_dict(checkpoint_encoder['state_dict'])
      optim_encoder.load_state_dict(checkpoint_encoder['optimizer'])

      checkpoint_decoder_R=torch.load(args.model_save_dir+"/decoder_R.pth")
      start_epoch=checkpoint_decoder_R['epoch']
      gen_decoder_R.load_state_dict(checkpoint_decoder_R['state_dict'])
      optim_decoder_R.load_state_dict(checkpoint_decoder_R['optimizer'])

      checkpoint_decoder_S=torch.load(args.model_save_dir+"/decoder_S.pth")
      start_epoch=checkpoint_decoder_S['epoch']
      gen_decoder_S.load_state_dict(checkpoint_decoder_S['state_dict'])
      optim_decoder_S.load_state_dict(checkpoint_decoder_S['optimizer'])

    # --------------------------------------------------------------------------------
    # c gen_encoder_param: print number of parameters of network
    gen_encoder_param=print_network(gen_encoder)
    gen_decoder_R_param=print_network(gen_decoder_R)
    gen_decoder_S_param=print_network(gen_decoder_S)
    entire_param=gen_encoder_param+gen_decoder_R_param+gen_decoder_S_param
    print("gen_encoder:",gen_encoder_param)
    print("gen_decoder_R:",gen_decoder_R_param)
    print("gen_decoder_S:",gen_decoder_S_param)
    print("Entire:",entire_param)

    # --------------------------------------------------------------------------------
    # c returns network, optimizer, schedulers
    if args.scheduler=="cyclic" or \
        args.scheduler=="cosine" or \
        args.scheduler=="adamW_cosine":
      return gen_encoder,gen_decoder_R,gen_decoder_S,\
              optim_encoder,optim_decoder_R,optim_decoder_S,\
              scheduler_encoder,scheduler_decoder_R,scheduler_decoder_S
    else:
      return gen_encoder,gen_decoder_R,gen_decoder_S,\
              optim_encoder,optim_decoder_R,optim_decoder_S

def save_checkpoint(state,filename):
    torch.save(state,filename)

def denorm(x):
    out=(x+1)/2
    return out.clamp(0,1)

def remove_existing_gradients_before_starting_new_training(
  model_list,args):

  if args.use_integrated_decoders=="True":
    if args.scheduler=="cyclic":
      model_list["gen_net"].zero_grad()
      # scheduler_encoder.batch_step()
    elif args.scheduler=="cosine":
      model_list["optimizer"].zero_grad()
      # scheduler_encoder.step()
    else:
      model_list["gen_net"].zero_grad()
  else:
    if args.scheduler=="cyclic":
      model_list["optim_encoder"].zero_grad()
      model_list["optim_decoder_R"].zero_grad()
      model_list["optim_decoder_S"].zero_grad()
      # scheduler_encoder.batch_step()
      # scheduler_decoder_R.batch_step()
      # scheduler_decoder_S.batch_step()
    elif args.scheduler=="cosine":
      model_list["optim_encoder"].zero_grad()
      model_list["optim_decoder_R"].zero_grad()
      model_list["optim_decoder_S"].zero_grad()
      # scheduler_encoder.step()
      # scheduler_decoder_R.step()
      # scheduler_decoder_S.step()
    else:
      model_list["optim_encoder"].zero_grad()
      model_list["optim_decoder_R"].zero_grad()
      model_list["optim_decoder_S"].zero_grad()

def save_trained_model_after_training_batch(one_ep,bs,model_list,args):
  if args.use_integrated_decoders=="True":
    batch_size=int(args.batch_size)
    gen_net=model_list["gen_net"]
    optimizer=model_list["optimizer"]

    try:
      # Save model every batch affected by leaping
      if bs%(batch_size*int(args.leaping_batchsize_for_saving_model))==0:
        encoder_path=args.model_save_dir+"/gen_net_batch_"+str(bs)+".pth"
        save_checkpoint(
          {'epoch':one_ep+1,
           'state_dict':gen_net.state_dict(),
           'optimizer':optimizer.state_dict()},
          encoder_path)
    except:
      print(traceback.format_exc())
      print("You may have passed 0")

  else:
    batch_size=int(args.batch_size)
    gen_encoder=model_list["gen_encoder"]
    optim_encoder=model_list["optim_encoder"]
    gen_decoder_R=model_list["gen_decoder_R"]
    optim_decoder_R=model_list["optim_decoder_R"]
    gen_decoder_S=model_list["gen_decoder_S"]
    optim_decoder_S=model_list["optim_decoder_S"]

    try:
      # Save model every batch affected by leaping
      if bs%(batch_size*int(args.leaping_batchsize_for_saving_model))==0:
        encoder_path=args.model_save_dir+"/encoder_batch_"+str(bs)+".pth"
        decoder_R_path=args.model_save_dir+"/decoder_R_batch_"+str(bs)+".pth"
        decoder_S_path=args.model_save_dir+"/decoder_S_batch_"+str(bs)+".pth"

        save_checkpoint(
          {'epoch':one_ep+1,
          'state_dict':gen_encoder.state_dict(),
          'optimizer':optim_encoder.state_dict()},
          encoder_path)
        save_checkpoint(
          {'epoch':one_ep+1,
          'state_dict':gen_decoder_R.state_dict(),
          'optimizer':optim_decoder_R.state_dict()},
          decoder_R_path)
        save_checkpoint(
          {'epoch':one_ep+1,
          'state_dict':gen_decoder_S.state_dict(),
          'optimizer':optim_decoder_S.state_dict()},
          decoder_S_path)
    except:
      print(traceback.format_exc())
      print("You may have passed 0")

def save_trained_model_after_training_epoch(one_ep,bs,model_list,args):
  if args.use_integrated_decoders=="True":
    # --------------------------------------------------------------------------------
    # Saveparameters'valuesatendofeachepoch
    gen_net=model_list["gen_net"]
    optimizer=model_list["optimizer"]

    net_path=args.model_save_dir+"/deep_shading_net_"+str(one_ep)+".pth"

    save_checkpoint(
      {'epoch':one_ep+1,
       'state_dict':gen_net.state_dict(),
       'optimizer':optimizer.state_dict()},
      net_path)

  else:
    # --------------------------------------------------------------------------------
    gen_encoder=model_list["gen_encoder"]
    optim_encoder=model_list["optim_encoder"]
    gen_decoder_R=model_list["gen_decoder_R"]
    optim_decoder_R=model_list["optim_decoder_R"]
    gen_decoder_S=model_list["gen_decoder_S"]
    optim_decoder_S=model_list["optim_decoder_S"]
    
    # Save parameters' values at end of each epoch
    encoder_path=args.model_save_dir+"/encoder_epoch_"+str(one_ep)+".pth"
    decoder_R_path=args.model_save_dir+"/decoder_R_epoch_"+str(one_ep)+".pth"
    decoder_S_path=args.model_save_dir+"/decoder_S_epoch_"+str(one_ep)+".pth"

    save_checkpoint(
      {'epoch':one_ep+1,
       'state_dict':gen_encoder.state_dict(),
       'optimizer':optim_encoder.state_dict()},
      encoder_path)
    save_checkpoint(
      {'epoch':one_ep+1,
       'state_dict':gen_decoder_R.state_dict(),
       'optimizer':optim_decoder_R.state_dict()},
      decoder_R_path)
    save_checkpoint(
      {'epoch':one_ep+1,
       'state_dict':gen_decoder_S.state_dict(),
       'optimizer':optim_decoder_S.state_dict()},
      decoder_S_path)

def empty_cache_of_gpu_after_training_batch():
  num_gpus=torch.cuda.device_count()
  for gpu_id in range(num_gpus):
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    # print("empty cache gpu in dense")

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """
        Resnet18, resnet34, resnet50, resnet101
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "vgg":
        """
        VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "densenet":
        """
        Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """
        Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size
