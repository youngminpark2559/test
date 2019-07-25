import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys,os,copy,argparse
import time,timeit,datetime
import glob,natsort
import cv2
from PIL import Image
from skimage import data, img_as_float
from skimage.util import random_noise
import scipy.misc
from skimage.viewer import ImageViewer
from random import shuffle
import traceback
import subprocess

# ================================================================================
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torch.autograd import gradcheck
import visdom

# ================================================================================
from src.loss_functions import loss_functions as loss_functions

from src.utils import utils_data_loss as utils_data_loss
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image
from src.utils import utils_net as utils_net
from src.utils import utils_data as utils_data

from src.utils_for_dataset import utils_dataset_skin_cancer as utils_dataset_skin_cancer
from src.utils_for_dataset import dataset_returning_one_pair_of_path as dataset_returning_one_pair_of_path

from src.hyper_parameters import set_hyper_parameters as set_hyper_parameters

from src.utils_for_analyzing_training_result import visualize_loss_values as visualize_loss_values

from src.model_api import model_api_module as model_api_module

# from src.tensorboard import logger as logger

# ================================================================================
def train(args):
  epoch=int(args.epoch)
  batch_size=int(args.batch_size)

  # ================================================================================
  if args.use_visdom=="True":
    global plotter
    plotter=visdom_module.VisdomLinePlotter(env_name='IID network plot')
    visdom_i=1
    # vis=visdom.Visdom(port=8097,server="http://localhost/")
    vis=visdom.Visdom(env="IID network plot")

  # ================================================================================
  df_train,df_val,df_test=utils_data.prepare_dataset(args)

  # ================================================================================
  model_name = 'densenet'
  num_classes = 7
  feature_extract = False

  # ================================================================================
  model_ft, input_size = utils_net.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

  # ================================================================================
  # Define the device:
  device = torch.device('cuda:0')

  # ================================================================================
  # Put the model on the device:
  model = model_ft.to(device)

  # ================================================================================
  norm_mean = (0.49139968, 0.48215827, 0.44653124)
  norm_std = (0.24703233, 0.24348505, 0.26158768)

  # ================================================================================
  # define the transformation of the train images.
  train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                          transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])

  # ================================================================================
  # define the transformation of the val images.
  val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),
                                      transforms.Normalize(norm_mean, norm_std)])

  # ================================================================================
  training_set = utils_dataset_skin_cancer.HAM10000(df_train, transform=train_transform)
  train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=4)

  # ================================================================================
  validation_set=utils_dataset_skin_cancer.HAM10000(df_val, transform=train_transform)
  val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=4)

  # ================================================================================
  # we use Adam optimizer, use cross entropy loss as our loss function
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  # ================================================================================
  if args.use_focal_loss=="True":
    criterion=utils_data_loss.focal_loss
  else:
    criterion=nn.CrossEntropyLoss().to(device)
 
  # ================================================================================
  total_loss_train, total_acc_train = [],[]
  train_loss_record=[]
  validate_loss_record=[]

  # ================================================================================
  def perform_train(train_loader, model, criterion, optimizer, epoch):
      model.train()

      # ================================================================================
      train_loss = utils_data_loss.AverageMeter()
      train_acc = utils_data_loss.AverageMeter()

      # ================================================================================
      curr_iter = (epoch - 1) * len(train_loader)

      # ================================================================================
      for i, data in enumerate(train_loader):
          images, labels = data
          # print("images",images.shape)
          # print("labels",labels)
          # images torch.Size([32, 3, 224, 224])
          # labels tensor([4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 2, 1, 4, 1, 4, 4, 6, 4, 2, 2, 4, 4, 2, 3, 2, 2, 4, 4, 6, 5, 6, 1])

          # ================================================================================
          if args.use_focal_loss=="True":
            labels=utils_data_loss.one_hot_embedding(labels, num_classes=7)

          # ================================================================================
          N = images.size(0)

          # ================================================================================
          # print('image shape:',images.size(0), 'label shape',labels.size(0))

          # ================================================================================
          images = Variable(images).to(device)
          labels = Variable(labels).to(device)
          # print("labels",labels)
          # tensor([6, 4, 4, 1, 2, 6, 3, 1, 4, 2, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 4, 4, 4, 4, 2], device='cuda:0')

          # ================================================================================
          optimizer.zero_grad()

          # ================================================================================
          outputs = model(images)
          # print("outputs",outputs)
          # tensor([[-0.8565, -0.1669,  0.7589, -0.4439, -0.2993, -0.0306, -0.8193],
          #         [-0.0700,  0.3807,  0.4200, -0.0492, -0.2753, -0.2942, -0.7149],
          #         [-0.5549,  0.7094,  1.0089, -0.1880,  0.2004, -1.1695, -0.4370],

          # print("outputs",outputs.shape)
          # torch.Size([32, 7])

          # ================================================================================
          # loss = criterion(outputs, labels)
          # print("loss",loss)
          # tensor(1.9242, device='cuda:0', grad_fn=<NllLossBackward>)

          loss = criterion(outputs, labels)
          print("loss",loss)
          # tensor(nan, device='cuda:0', grad_fn=<MeanBackward1>)

          # ================================================================================
          loss.backward()

          # ================================================================================
          optimizer.step()

          # ================================================================================
          prediction = outputs.max(1, keepdim=True)[1]

          # ================================================================================
          if args.use_focal_loss=="True":
            backed_to_index=labels.max(1,keepdim=True)[1]
            train_acc.update(prediction.eq(backed_to_index.view_as(prediction)).sum().item()/N)
          else:
            train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)

          # ================================================================================
          train_loss.update(loss.item())
          train_loss_record.append(loss.item())

          # ================================================================================
          curr_iter += 1

          # ================================================================================
          if (i + 1) % 100 == 0:
              print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))
              total_loss_train.append(train_loss.avg)
              total_acc_train.append(train_acc.avg)

      # ================================================================================
      return train_loss.avg, train_acc.avg

  # ================================================================================
  def validate(val_loader, model, criterion, optimizer, epoch):
      model.eval()
      val_loss = utils_data_loss.AverageMeter()
      val_acc = utils_data_loss.AverageMeter()
      with torch.no_grad():
          for i, data in enumerate(val_loader):
              images, labels = data

              # ================================================================================
              if args.use_focal_loss=="True":
                labels=utils_data_loss.one_hot_embedding(labels, num_classes=7)
              
              # ================================================================================
              N = images.size(0)
              images = Variable(images).to(device)
              labels = Variable(labels).to(device)

              outputs = model(images)
              prediction = outputs.max(1, keepdim=True)[1]

              # ================================================================================
              if args.use_focal_loss=="True":
                backed_to_index=labels.max(1,keepdim=True)[1]
                val_acc.update(prediction.eq(backed_to_index.view_as(prediction)).sum().item()/N)
              else:
                val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)

              # ================================================================================
              vali_loss=criterion(outputs, labels).item()
              validate_loss_record.append(vali_loss)
              
              # ================================================================================
              val_loss.update(vali_loss)

      print('------------------------------------------------------------')
      print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
      print('------------------------------------------------------------')
      # [epoch 10], [iter 100 / 157], [train loss 0.02759], [train acc 0.85094]
      # [epoch 10], [iter 100 / 157], [train loss 0.39521], [train acc 0.85344]

      return val_loss.avg, val_acc.avg

  # ================================================================================
  epoch_num = 10
  # epoch_num = 2
  best_val_acc = 0
  total_loss_val, total_acc_val = [],[]

  # ================================================================================
  for epoch in range(1, epoch_num+1):
      loss_train, acc_train = perform_train(train_loader, model, criterion, optimizer, epoch)
      loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch)
      total_loss_val.append(loss_val)
      total_acc_val.append(acc_val)
      if acc_val > best_val_acc:
          best_val_acc = acc_val
          print('*****************************************************')
          print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
          print('*****************************************************')
          # [epoch 10], [val loss 0.06447], [val acc 0.81746]
          # [epoch 10], [val loss 1.01743], [val acc 0.81597]

  # ================================================================================
  utils_data_loss.loss_visualization(train_loss_record,validate_loss_record,epoch_num)

  # ================================================================================
  print("train_loss_record",train_loss_record)
  print("validate_loss_record",validate_loss_record)
