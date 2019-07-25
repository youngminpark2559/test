# conda activate py36gputorch100 && cd /mnt/1T-5e7/Companies/Colab_nutri/Skin/prj_root

# ================================================================================
# Train
# - without prune
# rm e.l && python main.py \
# --train_mode=True \
# --batch_size=3 \
# --epoch=20 \
# --use_saved_model=False \
# --model_save_dir="./ckpt" \
# --model_file_name_when_saving_and_loading_model="/custom_IID.pth" \
# --use_local_piecewise_constant_loss=True \
# --use_augmentor=False \
# --use_integrated_decoders=True \
# --use_loss_display=True \
# --input_size=256 \
# --leaping_batchsize_for_saving_model=1 \
# --use_multi_gpu=False \
# --text_file_for_paths_dir="/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/kaggle.com_saketc_skin-lesion-analyser-sla/base_dir/train_dir" \
# --iiw_training_strategy="batch" \
# --check_input_output_via_multi_gpus=False \
# --measure_train_time=True \
# --scheduler=None \
# --train_method="basic_cnn" \
# --use_visdom="False" \
# --save_imgs="True" \
# --visualize_loss_values="False" \
# --use_focal_loss="False" \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
# Options

# --iiw_training_strategy="pixbypix" \
# --iiw_training_strategy="batch" \

# --text_file_for_paths_dir="/mnt/1T-5e7/image/whole_dataset" \
# --text_file_for_paths_dir="/mnt/1T-5e7/image/whole_dataset/text_for_colab/real/temp/small" \

# --use_saved_model=False \
# --use_saved_model=True \

# --train_method="separately_update_over_full_dataset_using_custom_dataset_class" \
# --train_method="train_with_prune_filling_0s" \

# --save_imgs="False" \
# --save_imgs="True" \

# ================================================================================
# Inference mode

# rm e.l && python main.py \
# --train_mode=False \
# --batch_size=2 \
# --epoch=1 \
# --use_saved_model=True \
# --model_save_dir="./ckpt" \
# --model_file_name_when_saving_and_loading_model="/custom_IID.pth" \
# --use_local_piecewise_constant_loss=True \
# --use_augmentor=True \
# --use_integrated_decoders=True \
# --use_loss_display=True \
# --input_size=256 \
# --leaping_batchsize_for_saving_model=1 \
# --use_multi_gpu=False \
# --text_file_for_paths_dir="/mnt/1T-5e7/image/whole_dataset/text_for_colab/real/temp" \
# --iiw_training_strategy="batch" \
# --check_input_output_via_multi_gpus=False \
# --measure_train_time=True \
# --scheduler=None \
# --train_method="separately_update_over_full_dataset_using_custom_dataset_class" \
# --use_visdom="False" \
# --visualize_loss_values="False" \
# 2>&1 | tee -a e.l && code e.l

# ================================================================================
import datetime
import sys,os,copy,argparse
# current working directory /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root
# path of current file /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/main.py
# dir name of current file /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root

import torch

# ================================================================================
from src.base_class import base_class_for_argument as base_class_for_argument

from src.argument_api import argument_api_module as argument_api_module

from src.train import basic_cnn as basic_cnn

# ================================================================================
def start_func(args):
  if args.train_method=="basic_cnn":
    basic_cnn.train(args)

# ================================================================================
if __name__=="__main__":
  # c argument_api: instance of Argument_API_class
  argument_api=argument_api_module.Argument_API_class()

  # c args: member attribute from argument_api
  args=argument_api.args
  # print("args",args)
  # Namespace(
  #   batch_size='2',
  #   check_input_output_via_multi_gpus='False',
  #   epoch='2',
  #   input_size='256',
  #   iteration=4,
  #   leaping_batchsize_for_saving_model='1',
  #   measure_train_time='True',
  #   model_file_name_when_saving_and_loading_model='/custom_IID.pth--use_local_piecewise_constant_loss=True',
  #   model_save_dir='./ckpt',
  #   scheduler='None',
  #   seed=42,
  #   text_file_for_paths_dir='/mnt/1T-5e7/image/whole_dataset',
  #   train_dir=None,
  #   train_method='separately_update_over_full_dataset_using_custom_dataset_class',
  #   train_mode='True',
  #   trained_network_path=None,
  #   use_augmentor='True',
  #   use_integrated_decoders='True',
  #   use_local_piecewise_constant_loss=False,
  #   use_loss_display='True',
  #   use_multi_gpu='False',
  #   use_pretrained_resnet152=False,
  #   use_saved_model='False')
  
  # ================================================================================
  start_func(args)
