import sys,os,copy,argparse

import torch
import torch.nn as nn

# ================================================================================
from src.utils import utils_net as utils_net

from src.networks import networks as networks

# ================================================================================
class Model_API_class():
  def __init__(self,args):
    super(Model_API_class).__init__()

    self.lr=0.0001
    self.args=args

    if self.args.use_integrated_decoders=="True":
      self.gen_net,self.optimizer=self.net_generator()
    else:
      self.gen_encoder,self.gen_decoder_R,self.gen_decoder_S,self.optim_encoder,self.optim_decoder_R,self.optim_decoder_S=self.net_generator()

  def gpu_setting(self,generated_model):
    gen_net=generated_model

    # ================================================================================
    if self.args.use_multi_gpu=="True":
      num_gpu=torch.cuda.device_count()
      print("num_gpu",num_gpu)

      DEVICE_IDS=list(range(num_gpu))
      gen_net=nn.DataParallel(gen_net,device_ids=DEVICE_IDS)

    else: # single GPU
      num_gpu=torch.cuda.device_count()
      print("num_gpu",num_gpu)
      DEVICE_IDS=list(range(num_gpu))
      gen_net=nn.DataParallel(gen_net,device_ids=DEVICE_IDS)
    return gen_net

  def load_checkpoint_file(self,generated_network,optimizer):
    gen_net=generated_network
    optimizer=optimizer

    # ================================================================================
    if self.args.use_saved_model=="True":
      path=self.args.model_save_dir+self.args.model_file_name_when_saving_and_loading_model
      checkpoint_gener_direct_rgb=torch.load(path)

      gen_net.load_state_dict(checkpoint_gener_direct_rgb['state_dict'])
      optimizer.load_state_dict(checkpoint_gener_direct_rgb['optimizer'])

    else: # use_saved_model is False
      pass

    return gen_net,optimizer

  def net_generator(self):
    gen_net=networks.Pretrained_ResNet50()

    # ================================================================================
    # Configure GPU
    gen_net=self.gpu_setting(gen_net)

    # ================================================================================
    # Configure optimizer and scheduler
    optimizer=torch.optim.Adam(gen_net.parameters(),lr=self.lr)

    # ================================================================================
    # Load trained model and optimizer
    gen_net,optimizer=self.load_checkpoint_file(generated_network=gen_net,optimizer=optimizer)

    # ================================================================================
    gen_net.cuda()

    # ================================================================================
    for state in optimizer.state.values():
      for k, v in state.items():
        if isinstance(v,torch.Tensor):
          state[k]=v.cuda()

    # ================================================================================
    # Print network infomation
    gen_net_param=self.print_network(gen_net)
    print("gen_net:",gen_net_param)
    # gen_net: 23510081

    # ================================================================================
    return gen_net,optimizer
    
  def print_network(self,gen_net,model_structure_display=False):
    
    if model_structure_display==True:
      print(gen_net)

    # ================================================================================
    # Initialize number of parameters by 0
    num_params=0

    # Iterate all parameters of net
    for param in gen_net.parameters():
      # Increment number of parameter
      num_params+=param.numel()
    
    return num_params

  def save_checkpoint(self,state,filename):
    torch.save(state,filename)

  def remove_existing_gradients_before_starting_new_training(self):
    if self.args.use_integrated_decoders=="True":
      self.gen_net.zero_grad()
    else:
      self.gen_encoder.zero_grad()
      self.gen_decoder_R.zero_grad()
      self.gen_decoder_S.zero_grad()

  def save_model(self,is_batch,iter_num):
    # @ If you use integrated decoders net
    if self.args.use_integrated_decoders=="True":
      # @ Create directory if it doesn't exist
      if not os.path.exists(self.args.model_save_dir):
        os.makedirs(self.args.model_save_dir)

      # @ c model_name: name of model
      model_name=self.args.model_file_name_when_saving_and_loading_model.split(".")[0]

      # @ c net_path: path of model
      if is_batch==True:
        net_path=self.args.model_save_dir+model_name+"_batch_"+str(iter_num)+".pth"
      else:
        net_path=self.args.model_save_dir+model_name+"_epoch_"+str(iter_num)+".pth"

      # @ save model
      self.save_checkpoint(
        state={'state_dict':self.gen_net.state_dict(),'optimizer':self.optimizer.state_dict()},
        filename=net_path)

    else: # use_integrated_decoders is False
      pass

  def empty_cache_of_gpu_after_training_batch(self):
    # c num_gpus: num of GPUs
    num_gpus=torch.cuda.device_count()

    # Iterate all GPUs
    for gpu_id in range(num_gpus):

      # Set GPU
      torch.cuda.set_device(gpu_id)

      # Empty GPU
      torch.cuda.empty_cache()
