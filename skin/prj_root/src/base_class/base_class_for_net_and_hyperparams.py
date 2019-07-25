


# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/base_class/base_class_for_net_and_hyperparams.py

# ================================================================================
from src.utils import utils_net as utils_net

# ================================================================================
class Base_For_Net_And_Params(object):

  def __init__(self,args,num_entire_imgs):
    
    # ================================================================================
    # Set arguments
    self.args=args

    # ================================================================================
    # Set hyperparameters
    self.w_dense_loss_data_S=20000000.0
    self.w_dense_loss_data_R=20000000.0
    self.w_dense_loss_data_I=20000000.0
    self.w_dense_loss_grad_S=5.0
    self.w_dense_loss_grad_R=5.0
    self.w_dense_loss_grad_I=5.0

    self.w_cgmit_loss_data_S=500.0
    self.w_cgmit_loss_data_R=500.0
    self.w_cgmit_loss_data_I=500.0
    self.w_cgmit_loss_grad_R=50.0
    self.w_cgmit_loss_grad_S=50.0
    self.w_cgmit_loss_grad_I=50.0

    self.w_rsmooth_loss_final=300000.0

    self.w_en_ob_iiw_loss=500.0
    self.w_saw_loss=0.5

    self.w_bigtime_loss=0.1

    self.wd=0

    # ================================================================================
    # Set network
    self.model_list={}

    if self.args.use_integrated_decoders=="True":
      gen_net,optimizer=utils_net.net_generator(self.args,num_entire_imgs)
      self.model_list["gen_net"]=gen_net
      self.model_list["optimizer"]=optimizer
    
    else: # self.args.use_integrated_decoders=="False":
      if self.args.scheduler=="cyclic" or \
         self.args.scheduler=="cosine" or \
         self.args.scheduler=="adamW_cosine":
        gen_encoder,gen_decoder_R,gen_decoder_S,\
        optim_encoder,optim_decoder_R,optim_decoder_S,\
        scheduler_encoder,scheduler_decoder_R,scheduler_decoder_S=\
          utils_net.net_generator(self.args,num_entire_imgs)

        self.model_list["gen_encoder"]=gen_encoder
        self.model_list["gen_decoder_R"]=gen_decoder_R
        self.model_list["gen_decoder_S"]=gen_decoder_S
        self.model_list["optim_encoder"]=optim_encoder
        self.model_list["optim_decoder_R"]=optim_decoder_R
        self.model_list["optim_decoder_S"]=optim_decoder_S
        self.model_list["scheduler_encoder"]=scheduler_encoder
        self.model_list["scheduler_decoder_R"]=scheduler_decoder_R
        self.model_list["scheduler_decoder_S"]=scheduler_decoder_S
      
      else: # When you use adamW or ordinary optimizer (like adam, SGD)
        gen_encoder,gen_decoder_R,gen_decoder_S,\
        optim_encoder,optim_decoder_R,optim_decoder_S=\
          utils_net.net_generator(self.args,num_entire_imgs)

        self.model_list["gen_encoder"]=gen_encoder
        self.model_list["gen_decoder_R"]=gen_decoder_R
        self.model_list["gen_decoder_S"]=gen_decoder_S
        self.model_list["optim_encoder"]=optim_encoder
        self.model_list["optim_decoder_R"]=optim_decoder_R
        self.model_list["optim_decoder_S"]=optim_decoder_S
