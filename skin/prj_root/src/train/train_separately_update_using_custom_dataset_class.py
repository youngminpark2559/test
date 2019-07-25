import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys,os,copy,argparse
import time,timeit,datetime
import glob,natsort
import cv2
from PIL import Image
from skimage.transform import resize
from skimage.restoration import (denoise_tv_chambolle,denoise_bilateral,
                                 denoise_wavelet,estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
import scipy.misc
from skimage.viewer import ImageViewer
from random import shuffle
import gc
import Augmentor
import traceback
import subprocess
# import horovod.torch as hvd

# ================================================================================
import torch
import torch.nn as nn
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

from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image
from src.utils import utils_net as utils_net
from src.utils import utils_data as utils_data

from src.utils_for_dataset import dataset_returning_one_pair_of_path as dataset_returning_one_pair_of_path
from src.utils_for_dataset import get_batch_size_paths_using_iter as get_batch_size_paths_using_iter
from src.utils_for_dataset import utils_data_common_interface as utils_data_common_interface


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
  # @ Create network and optimizer
  
  model_api_inst=model_api_module.Model_API_class(args)

  # ================================================================================
  # @ c hyper_param_inst: instance of hyper parameter class
  
  hyper_param_inst=set_hyper_parameters.Set_h_p(args)

  # ================================================================================
  # @ For graph illustration of loss values

  lists_for_visualizing_losses=visualize_loss_values.Visualize_Loss_Values()

  # ================================================================================
  # @ c path_of_text_files
  path_of_text_files=text_file_api_module.Path_Of_Text_Files(args)

  # c txt_paths_in_dict: path of text files in dictionary
  txt_paths_in_dict=path_of_text_files.__dict__
  # print("txt_paths_in_dict",txt_paths_in_dict)

  # ================================================================================
  # # @ Check exception for batch size
  # (This is commented out because Pytorch's DataSet and DataLoader can manage this potential issue)

  # cgmit=len(dataset_dense_inst)

  # assert cgmit==len(dataset_iiw_inst),"The number of 'cgmit images' and 'iiw images' is different"

  # print("Current number of 'cgmit images' and 'iiw images':",cgmit)
  # print("Current cgmit/iiw batch size:",batch_size)
  # print("Possible cgmit/iiw batch size:",list(utils_common.divisorGenerator(cgmit)))
  
  # assert str(cgmit/batch_size).split(".")[-1]==str(0),"Check batch size"
  
  # ================================================================================
  # @ Grad CAM (tested visualization on gradient on trained network)

  # cam=utils_visualize_gradients.CAM(gen_decoder_R)
  # gen_decoder_S.right_conv_6.register_forward_hook(utils_hook_functions.printnorm)
  # gen_decoder_S.right_conv_6.register_backward_hook(utils_hook_functions.printgradnorm)

  # ================================================================================
  # @ Measure training time

  if args.measure_train_time=="True":
    start=timeit.default_timer()
  else:
    pass

  # ================================================================================
  if args.train_mode=="True":
    # @ Iterates all epochs
    for one_ep in range(epoch):

      # ================================================================================
      # c dataset_dense_inst: dataset instance of dense dataset
      dataset_dense_inst=dataset_returning_one_pair_of_path.Dataset_Returning_One_Pair_Of_Path(
        txt_paths_in_dict["cgintrinsic_trn"],txt_paths_in_dict["cgintrinsic_rgt"],args)
      
      dataset_dense_iter=iter(dataset_dense_inst)

      # ================================================================================
      dataset_tinghuiz_inst=dataset_returning_one_pair_of_path.Dataset_Returning_One_Pair_Of_Path(
        txt_paths_in_dict["tinghuiz_trn"],txt_paths_in_dict["tinghuiz_rgt"],args)

      dataset_tinghuiz_iter=iter(dataset_tinghuiz_inst)
      
      # ================================================================================
      # In my test, CGINTRINSIC + Tinghuiz + IIW was not that good
      # I thinks it's because CGINTRINSIC + Tinghuiz and IIW are too diferent in its format of dataset
      # This joint training was tested by being inspired by CGINTRINSICS project
      # Original code: https://github.com/lixx2938/CGIntrinsics

      # c dataset_iiw_inst: instance of dataset iiw
      # dataset_iiw_inst=dataset_returning_one_pair_of_path.Dataset_Returning_One_Pair_Of_Path(
      #   txt_paths_in_dict["iiw_trn"],txt_paths_in_dict["iiw_json"],args)

      # dataset_iiw_iter=iter(dataset_iiw_inst)

      # ================================================================================
      # Tested joint training along with SAW dataset, but my test with SAW didn't create good result
      # Original code: https://github.com/lixx2938/CGIntrinsics

      # dataset_saw_inst=dataset_returning_one_pair_of_path.Dataset_Returning_One_Pair_Of_Path(
      #   txt_paths_in_dict["saw_trn"],txt_paths_in_dict["saw_npy"],args)
      
      # dataset_saw_iter=iter(dataset_saw_inst)

      # ================================================================================
      # BigTime data is designed to train IID network in unsupervised manners
      # Original code: https://github.com/lixx2938/unsupervised-learning-intrinsic-images

      # dataset_bigtime_inst=dataset_returning_one_pair_of_path.Dataset_Returning_One_Pair_Of_Path_non_shuffle(
      #   txt_paths_in_dict["bigtime_trn"],txt_paths_in_dict["bigtime_mask"],args)

      # dataset_bigtime_iter=iter(dataset_bigtime_inst)

      # ================================================================================
      # num_entire_imgs=len(dataset_dense_inst)+len(dataset_tinghuiz_inst)+len(dataset_iiw_inst)
      num_entire_imgs=len(dataset_dense_inst)+len(dataset_tinghuiz_inst)
      # print("num_entire_imgs",num_entire_imgs)
      # 16

      args.__setattr__("num_entire_imgs",num_entire_imgs)
      
      # ================================================================================
      num_iteration_for_iter=int(len(dataset_dense_inst)/int(args.batch_size))

      print("args.leaping_batchsize_for_saving_model)",args.leaping_batchsize_for_saving_model)
      print("num_iteration_for_iter",num_iteration_for_iter)

      if int(args.leaping_batchsize_for_saving_model)==num_iteration_for_iter or int(args.leaping_batchsize_for_saving_model)>num_iteration_for_iter:
        raise Exception("args.leaping_batchsize_for_saving_model must be smaller than num_iteration_for_iter")
      else:
        pass

      # ================================================================================
      # If you don't use Augmentor
      if args.use_augmentor=="False":

        # c dense_d_bs_pa: dense dataset batch size paths
        dense_d_bs_pa=dense_d_list[bs:bs+batch_size]
        cgmit_d_bs_pa=cgmit_d_list[bs:bs+batch_size]

        # ================================================================================
        dense_trn_imgs,dense_rgt_imgs=utils_common.generate_dense_dataset(
          dense_d_bs_pa,img_h=512,img_w=512)

        trn_imgs_dense,rgt_imgs_dense,sgt_imgs_dense,mask_imgs_dense=\
          utils_common.generate_cgmit_dataset(cgmit_d_bs_pa,img_h=512,img_w=512)

      else: # If you use Augmentor
        # @ Iterate all entire images during single epoch
        for one_iter in range(num_iteration_for_iter):
          # @ Remove gradients
          model_api_inst.remove_existing_gradients_before_starting_new_training()

          # ================================================================================
          # @ Train over dense dataset

          loss_r_data_dense,loss_s_data_dense,loss_o_data_dense,loss_o_grad_dense,loss_s_grad_dense,loss_r_grad_dense,pred_r_color_dense,pred_s_dense=\
            train_over_dense_dataset.train(
              dataset_dense_iter,batch_size,model_api_inst,hyper_param_inst,lists_for_visualizing_losses,args)

          # ================================================================================
          # @ Train over tinghuiz dataset

          loss_r_data_tinghuiz,loss_s_data_tinghuiz,loss_o_data_tinghuiz,loss_o_grad_tinghuiz,loss_s_grad_tinghuiz,loss_r_grad_tinghuiz,tinghuiz_r,tinghuiz_s=\
            train_over_tinghuiz_dataset.train(
              dataset_tinghuiz_iter,batch_size,model_api_inst,hyper_param_inst,lists_for_visualizing_losses,args)

          # ================================================================================
          # @ Train over IIW dataset

          # iiw_I_loss
          # loss_iiw=train_over_iiw_dataset.train(
          #   dataset_iiw_iter,batch_size,model_api_inst,hyper_param_inst,lists_for_visualizing_losses,args)
          # print("loss_iiw",loss_iiw)
          # tensor([0.0001], device='cuda:0', grad_fn=<MulBackward0>)
          # 3.633527994155884

          # Took large time
          # 0:02:10.749899
          
          # After removing "for loop in data augmentation", small time
          # 0:00:00.720973

          # After one-image-by-one-image traing, more small time
          # 0:00:00.405261

          # ================================================================================
          # @ Train over SAW dataset
          
          # saw_I_loss
          # saw_loss,saw_I_loss=train_over_saw_dataset.train(
          #   dataset_saw_iter,batch_size,model_api_inst,hyper_param_inst,lists_for_visualizing_losses,args)
          # print("saw_loss",saw_loss)
          # 1.520530343055725

          # ================================================================================
          # Update based on bigtime dataset (BT)

          # bigtime_loss_I
          # bigtime_loss_r,bigtime_loss_I=train_over_bigtime_dataset.train(
          #   dataset_bigtime_iter,batch_size,model_api_inst,hyper_param_inst,lists_for_visualizing_losses,args)
          # print("bigtime_loss_I",bigtime_loss_I)
          # print("bigtime_loss_r",bigtime_loss_r)

          # ================================================================================
          entire_loss=loss_r_data_dense+loss_s_data_dense+loss_o_data_dense+loss_o_grad_dense+loss_s_grad_dense+loss_r_grad_dense+\
                      loss_r_data_tinghuiz+loss_s_data_tinghuiz+loss_o_data_tinghuiz+loss_o_grad_tinghuiz+loss_s_grad_tinghuiz+loss_r_grad_tinghuiz

          entire_loss.backward()
          model_api_inst.optimizer.step()
          model_api_inst.empty_cache_of_gpu_after_training_batch()

          # ================================================================================
          # region @ Save loss into tensorboard 
          # (I tested tensorboard but this had been commented out because I can plot losses and predicted image by using visdom)

          # --------------------------------------------------------------------------------
          # conda activate py36gputorch100 &&
          # cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/tensorboard &&
          # tensorboard --logdir='./logs' --port=6006

          # --------------------------------------------------------------------------------
          # tb_logger=logger.Logger('./src/tensorboard/logs')

          # --------------------------------------------------------------------------------
          # print(lists_for_visualizing_losses.lo_list)
          # [('loss_r_data_dense',[75.04694366455078]),
          #  ('loss_r_grad_dense',[1.0475735664367676]),
          #  ('loss_s_data_dense',[48.632972717285156]),
          #  ('loss_s_grad_dense',[0.5619246363639832]),
          #  ('loss_o_data_dense',[46.09053039550781]),
          #  ('loss_o_grad_dense',[1.4934182167053223]),
          #  ('iiw_loss_temp',[2.321777105331421]),
          #  ('saw_loss_temp',[0.8101050853729248]),
          #  ('bigtime_loss_temp',[0.346628874540329])]

          # --------------------------------------------------------------------------------
          # aaa,bbb,ccc,ddd,eee,fff,ggg,hhh,iii=lists_for_visualizing_losses.lo_list
          # print("a",a)
          # a ('loss_r_data_dense', [82.376953125])

          # --------------------------------------------------------------------------------
          # info={aaa[0]:aaa[1][0],bbb[0]:bbb[1][0],ccc[0]:ccc[1][0],ddd[0]:ddd[1][0],eee[0]:eee[1][0],
          #       fff[0]:fff[1][0],ggg[0]:ggg[1][0],hhh[0]:hhh[1][0],iii[0]:iii[1][0]}

          # --------------------------------------------------------------------------------
          # for tag,value in info.items():
          #   tb_logger.scalar_summary(tag,value,int(one_iter)+1)
          # endregion

          # ================================================================================
          # region @ Use visdom
          if args.use_visdom=="True":
            # pip install visdom
            # conda install -c conda-forge visdom
            # python -m visdom.server
            # http://localhost:8097
            # <your_remote_server_ip>:<visdom port (default 8097)>
            plotter.plot("loss_r_data_dense","train","loss_r_data_dense",visdom_i,loss_r_data_dense.item())
            plotter.plot("loss_s_data_dense","train","loss_s_data_dense",visdom_i,loss_s_data_dense.item())
            plotter.plot("loss_o_data_dense","train","loss_o_data_dense",visdom_i,loss_o_data_dense.item())
            plotter.plot("loss_o_grad_dense","train","loss_o_grad_dense",visdom_i,loss_o_grad_dense.item())
            plotter.plot("loss_s_grad_dense","train","loss_s_grad_dense",visdom_i,loss_s_grad_dense.item())
            plotter.plot("loss_r_grad_dense","train","loss_r_grad_dense",visdom_i,loss_r_grad_dense.item())
            plotter.plot("loss_r_data_tinghuiz","train","loss_r_data_tinghuiz",visdom_i,loss_r_data_tinghuiz.item())
            plotter.plot("loss_s_data_tinghuiz","train","loss_s_data_tinghuiz",visdom_i,loss_s_data_tinghuiz.item())
            plotter.plot("loss_o_data_tinghuiz","train","loss_o_data_tinghuiz",visdom_i,loss_o_data_tinghuiz.item())
            plotter.plot("loss_o_grad_tinghuiz","train","loss_o_grad_tinghuiz",visdom_i,loss_o_grad_tinghuiz.item())
            plotter.plot("loss_s_grad_tinghuiz","train","loss_s_grad_tinghuiz",visdom_i,loss_s_grad_tinghuiz.item())
            plotter.plot("loss_r_grad_tinghuiz","train","loss_r_grad_tinghuiz",visdom_i,loss_r_grad_tinghuiz.item())
            # plotter.plot("loss_iiw","train","loss_iiw",visdom_i,loss_iiw.item())
            
            # images
            vis.images(Variable(torch.tensor(pred_r_color_dense[:2,:,:,:])),opts=dict(title='Reflectance', caption='Reflectance.'),win='wazzup')
            vis.images(Variable(torch.tensor(pred_s_dense[:2,:,:,:])),opts=dict(title='Shading', caption='Shading.'),win='wazzup2')

            visdom_i=visdom_i+1
          # endregion 

          # ================================================================================
          del loss_r_data_dense,loss_s_data_dense,loss_o_data_dense,loss_o_grad_dense,loss_s_grad_dense,loss_r_grad_dense,\
              loss_r_data_tinghuiz,loss_s_data_tinghuiz,loss_o_data_tinghuiz,loss_o_grad_tinghuiz,loss_s_grad_tinghuiz,loss_r_grad_tinghuiz

          # ================================================================================
          # @ Save model after batch

          if int(one_iter)==0:
            pass
          elif one_iter%int(args.leaping_batchsize_for_saving_model)==0:
            model_api_inst.save_model(is_batch=True,iter_num=one_iter)
          else:
            pass
          
      # ================================================================================
      # @ Save model after epoch

      model_api_inst.save_model(is_batch=False,iter_num=one_ep)

      # ================================================================================
      # @ Time for single epoch

      if args.measure_train_time=="True":
        stop=timeit.default_timer()
        took_time_sec=int(stop-start)
        took_time_min=str(datetime.timedelta(seconds=took_time_sec))
        print("Took minutes until finishing epoch of "+str(one_ep)+\
              " with "+str(args.scheduler)+" scheduler:",took_time_min)
      else:
        pass
    
    # ================================================================================
    # @ Visualize loss at the end of training

    if args.visualize_loss_values=="True":
      loss_list=lists_for_visualizing_losses.lo_list
      for i,tup in enumerate(loss_list):
        plt.subplot(int(np.ceil(len(loss_list)/3)),3,i+1)
        plt.title(tup[0])
        plt.plot(tup[1])
      plt.savefig("loss.png")
      plt.show()
    else:
      pass
  
  # ================================================================================
  else: # @ Test trained model
    with torch.no_grad():

      # @ Set test directories
      dense_trn_img_p="./Data/Images_for_inference/*.png"
      # dense_trn_img_p="/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/test/*.png"
      # dense_trn_img_p="/mnt/1T-5e7/mycodehtml/data_col/cv/IID_f_w_w/iiw-dataset/office/*.png"
      # dense_trn_img_p="/home/young/Downloads/images/training/temp/*.jpg"
      # dense_trn_img_p="/home/young/Downloads/Sticheo_example_video/0_No_Movement/sample/FPS15/*.png"
      # dense_trn_img_p="/home/young/Downloads/Sticheo_example_video/1_Small_Movement/sample/original_frame_FPS15/*.png"
      # dense_trn_img_p="/home/young/Downloads/Sticheo_example_video/2_Medium_Movement/sample/original_frame_FPS15/*.png"

      # @ For video frames
      # r_images="/home/young/Downloads/Sticheo_example_video/0_No_Movement/sample/FPS15/Decomposed/R/*.png"
      # s_images="/home/young/Downloads/Sticheo_example_video/0_No_Movement/sample/FPS15/Decomposed/S/*.png"
      # r_images="/home/young/Downloads/Sticheo_example_video/1_Small_Movement/sample/original_frame_FPS15/R/*.png"
      # s_images="/home/young/Downloads/Sticheo_example_video/1_Small_Movement/sample/original_frame_FPS15/S/*.png"
      # r_images="/home/young/Downloads/Sticheo_example_video/2_Medium_Movement/sample/R/*.png"
      # s_images="/home/young/Downloads/Sticheo_example_video/2_Medium_Movement/sample/S/*.png"

      # @ Recovered directories
      # recovered_imgs="/home/young/Downloads/Sticheo_example_video/0_No_Movement/sample/FPS15/Decomposed/Recovered/*.png"
      # recovered_imgs="/home/young/Downloads/Sticheo_example_video/1_Small_Movement/sample/original_frame_FPS15/Recovered/*.png"
      # recovered_imgs="/home/young/Downloads/Sticheo_example_video/2_Medium_Movement/sample/Recovered/*.png"

      # ================================================================================
      # @ Following codes can enhance resolution and can adjust brightness and gamma level

      # r_img_li=utils_common.get_file_list(r_images)
      # s_img_li=utils_common.get_file_list(s_images)
      
      # for one_pair in list(zip(r_img_li,s_img_li)):
      #   # print("one_pair[0]",one_pair[0])
      #   # /home/young/Downloads/Sticheo_example_video/0_No_Movement/sample/FPS15/Decomposed/R/dinner_0001_predref.png

      #   recovered_fn=one_pair[0].replace("_predref","").replace("/R/","/Recovered/")

      #   one_r_img=utils_image.load_img(one_pair[0])/255.0
      #   one_s_img=utils_image.load_img(one_pair[1])/255.0
      #   one_s_img=one_s_img[:,:,np.newaxis]
      #   recovered_img=one_r_img*one_s_img

      #   # recovered_img=cv2.normalize(recovered_img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
      #   # print("np.min(recovered_img)",np.min(recovered_img))
      #   # print("np.max(recovered_img)",np.max(recovered_img))

      #   # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
      #   # kernel = np.array([[-0.1,-0.1,-0.1], [-0.1,0.9,-0.1], [-0.1,-0.1,-0.1]])
      #   # recovered_img = cv2.filter2D(recovered_img, -1, kernel)

      #   scipy.misc.imsave(recovered_fn,recovered_img)

      # # ================================================================================
      # @ Following codes can enhance resolution and can adjust brightness and gamma level

      # recovered_img_li=utils_common.get_file_list(recovered_imgs)

      # for one_img in recovered_img_li:
      #   fn=one_img.replace("/Recovered/","/Recovered_Sharpened_Edited_brightness_and_constrast/")
      #   img = cv2.imread(one_img, 1)[:,:,::-1]
      #   # kernel = np.array([[-1,-1,-1], [-1,5,-1], [-1,-1,-1]])
      #   kernel = np.array([[-1,-1,-1,-1,-1],
      #                      [-1,2,2,2,-1],
      #                      [-1,2,8,2,-1],
      #                      [-2,2,2,2,-1],
      #                      [-1,-1,-1,-1,-1]])/8.0
      #   im = cv2.filter2D(img, -1, kernel)

      #   brightness = 50
      #   contrast = 30
      #   img = np.int16(im)
      #   img = img * (contrast/127+1) - contrast + brightness
      #   img = np.clip(img, 0, 255)
      #   im = np.uint8(img)

      #   # img = cv2.imread('your path',0)
      #   # brt = 40
      #   # img[img < 255-brt] += brt    //change any value in the 2D list < max limit
      #   # cv2.imshow('img'+ img)

      #   scipy.misc.imsave(fn,im)

      # ================================================================================
      dense_tr_img_li=utils_common.get_file_list(dense_trn_img_p)
      iteration=int(len(dense_tr_img_li))
      num_imgs=len(dense_tr_img_li)

      # Iterates all train images
      for itr in range(iteration):
        fn=dense_tr_img_li[itr].split("/")[-1].split(".")[0]

        ol_dense_tr_i=utils_image.load_img(dense_tr_img_li[itr])/255.0
        ori_h=ol_dense_tr_i.shape[0]
        ori_w=ol_dense_tr_i.shape[1]

        # Test various input resolutions
        ol_dense_tr_i=utils_image.resize_img(ol_dense_tr_i,img_h=512,img_w=512)
        # ol_dense_tr_i=utils_image.resize_img(ol_dense_tr_i,img_h=1280,img_w=1280)
        # ol_dense_tr_i=utils_image.resize_img(ol_dense_tr_i,img_h=1536,img_w=1536)
        # ol_dense_tr_i=utils_image.resize_img(ol_dense_tr_i,img_h=1792,img_w=1792)
        # ol_dense_tr_i=utils_image.resize_img(ol_dense_tr_i,img_h=1920,img_w=1920)
        # ol_dense_tr_i=utils_image.resize_img(ol_dense_tr_i,img_h=2048,img_w=2048)

        # ================================================================================
        ol_dense_tr_i=ol_dense_tr_i.transpose(2,0,1)
        ol_dense_tr_i=ol_dense_tr_i[np.newaxis,:,:,:]
        con_O_img_arr=Variable(torch.Tensor(ol_dense_tr_i).cuda())

        # ================================================================================
        if args.use_integrated_decoders=="True":
          
          dense_pred_S_imgs_direct=model_api_inst.gen_net(con_O_img_arr)

          for one_pred_img in range(dense_pred_S_imgs_direct.shape[0]):
            # --------------------------------------------------------------------------------
            pred_one_S_img=dense_pred_S_imgs_direct[one_pred_img,:,:,:].unsqueeze(0)
            pred_one_S_img=torch.nn.functional.interpolate(
              pred_one_S_img,size=(ori_h,ori_w),scale_factor=None,mode='bilinear',align_corners=True)
            
            one_tr_img=con_O_img_arr[one_pred_img,:,:,:].unsqueeze(0)
            one_tr_img=torch.nn.functional.interpolate(
              one_tr_img,size=(ori_h,ori_w),scale_factor=None,mode='bilinear',align_corners=True)

            # --------------------------------------------------------------------------------
            one_tr_img=one_tr_img.detach().cpu().numpy().squeeze().transpose(1,2,0)
            pred_one_S_img=pred_one_S_img.detach().cpu().numpy().squeeze()
            # pred_one_S_img=np.clip(pred_one_S_img,0.0,1.3)
            # pred_one_S_img=(pred_one_S_img-np.min(pred_one_S_img))/np.ptp(pred_one_S_img)
            pred_one_S_img[pred_one_S_img==0.0]=0.001
            pred_one_S_img=pred_one_S_img[:,:,np.newaxis]

            pred_one_R_img=one_tr_img/pred_one_S_img
            print("pred_one_R_img",np.min(pred_one_R_img))
            print("pred_one_R_img",np.max(pred_one_R_img))

            pred_one_R_img=np.clip(pred_one_R_img,0.0,1.2)
            pred_one_R_img=(pred_one_R_img-np.min(pred_one_R_img))/np.ptp(pred_one_R_img)

            # Tested various combinations, for example,
            # option1: no normalize + colorize
            # option2: normalize + colorize + clip
            # option2: no normalize + colorize + clip
            # ...

            # pred_one_S_img=np.clip(pred_one_S_img,0.0,1.3)
            # pred_one_S_img=(pred_one_S_img-np.min(pred_one_S_img))/np.ptp(pred_one_S_img)

            # img=np.clip(pred_one_R_img,0.0,1.2)
            # img=(img-np.mean(img))/np.std(img)
            # pred_one_R_img=(img-np.min(img))/np.ptp(img)
            # pred_one_R_img=utils_image.bilateral_f(one_tr_img,pred_one_R_img)

            # pred_one_R_img=utils_image.colorize_tc(pred_one_R_img,one_tr_img)
            # pred_one_R_img=pred_one_R_img.detach().cpu().numpy().squeeze().transpose(1,2,0)
            # pred_one_S_img=pred_one_S_img.detach().cpu().numpy().squeeze()

            # print("pred_one_R_img",np.min(pred_one_R_img))
            # print("pred_one_R_img",np.max(pred_one_R_img))
            # print("pred_one_S_img",np.min(pred_one_S_img))
            # print("pred_one_S_img",np.max(pred_one_S_img))

            scipy.misc.imsave('./result/'+fn+'_predsha.png',pred_one_S_img.squeeze())
            scipy.misc.imsave('./result/'+fn+'_predref.png',pred_one_R_img)

            if args.measure_train_time=="True":
              stop=timeit.default_timer()
              time_for_inference=stop-start
            else:
              pass
        else:
          o_conv_8_avgpool,o_conv_7_avgpool,o_conv_6_avgpool,o_conv_5_avgpool,\
          o_conv_4_avgpool,o_conv_3_avgpool,o_conv_2_avgpool,o_conv_1_avgpool=gen_encoder(con_O_img_arr)

          dense_pred_R_imgs_direct=gen_decoder_R(
            o_conv_8_avgpool,o_conv_7_avgpool,o_conv_6_avgpool,o_conv_5_avgpool,\
            o_conv_4_avgpool,o_conv_3_avgpool,o_conv_2_avgpool,o_conv_1_avgpool)
          dense_pred_S_imgs_direct=gen_decoder_S(
            o_conv_8_avgpool,o_conv_7_avgpool,o_conv_6_avgpool,o_conv_5_avgpool,\
            o_conv_4_avgpool,o_conv_3_avgpool,o_conv_2_avgpool,o_conv_1_avgpool)
          
          # out=cam.get_cam(0)
          
          for one_pred_img in range(dense_pred_S_imgs_direct.shape[0]):
            # --------------------------------------------------------------------------------
            # Resizebytorch.nn.functional.interpolate()
            pred_one_R_img=dense_pred_R_imgs_direct[one_pred_img,:,:,:].unsqueeze(0)
            pred_one_R_img=torch.nn.functional.interpolate(
              pred_one_R_img,size=(ori_h,ori_w),scale_factor=None,mode='bilinear',align_corners=True)
            
            pred_one_S_img=dense_pred_S_imgs_direct[one_pred_img,:,:,:].unsqueeze(0)
            pred_one_S_img=torch.nn.functional.interpolate(
              pred_one_S_img,size=(ori_h,ori_w),scale_factor=None,mode='bilinear',align_corners=True)
            
            one_tr_img=con_O_img_arr[one_pred_img,:,:,:].unsqueeze(0)
            one_tr_img=torch.nn.functional.interpolate(
              one_tr_img,size=(ori_h,ori_w),scale_factor=None,mode='bilinear',align_corners=True)

            # --------------------------------------------------------------------------------
            pred_one_R_img=pred_one_R_img.detach().cpu().numpy().squeeze()
            one_tr_img=one_tr_img.detach().cpu().numpy().squeeze().transpose(1,2,0)

            # pred_one_R_img,sha=utils_image.colorize(pred_one_R_img,one_tr_img)
            # img=np.clip(pred_one_R_img,0.0,1.2)
            # img=(img-np.mean(img))/np.std(img)
            # pred_one_R_img=(img-np.min(img))/np.ptp(img)
            # pred_one_R_img=utils_image.bilateral_f(one_tr_img,pred_one_R_img)

            pred_one_S_img=pred_one_S_img.detach().cpu().numpy().squeeze()

            # pred_one_R_img=utils_image.colorize_tc(pred_one_R_img,one_tr_img)
            # pred_one_R_img=pred_one_R_img.detach().cpu().numpy().squeeze().transpose(1,2,0)
            # pred_one_S_img=pred_one_S_img.detach().cpu().numpy().squeeze()

            # print("pred_one_R_img",np.min(pred_one_R_img))
            # print("pred_one_R_img",np.max(pred_one_R_img))
            # print("pred_one_S_img",np.min(pred_one_S_img))
            # print("pred_one_S_img",np.max(pred_one_S_img))

            scipy.misc.imsave('./result/'+fn+'_predsha.png',pred_one_S_img)
            scipy.misc.imsave('./result/'+fn+'_predref.png',pred_one_R_img)

      print("input_size",args.input_size)      
      print("args.batchsize",args.batch_size)
      print("num_imgs",num_imgs)
      print("time_for_inference",time_for_inference)
      
      # input_size 512
      # args.batchsize 2
      # num_imgs 50
      # time_for_inference 14.74090022100063

      # 50 images is processed within 14 seconds (when it's 2 batch size)
      # 50/14=3.5 means 3.5 images can be processed within 1 second, FPS=3.5

      # If I use 3-times larger batchsize (like 6 batchsize), 14 seconds will be 14/3=4.6 seconds
      # It means 50 images will be processed within 4.6 seconds
      # 50/4.6=10.8 means 10.8 images can be processed within 1 second, FPS=10.8
      
      # 1280x1280
      # batch size 1
      # num_imgs 50
      # time_for_inference 20.96023238900011
      # 50/21, 2.38 FPS

