import sys,os,copy,argparse

class Set_h_p():
  def __init__(self,args):
    self.args=args
    self.weights_for_dataset=self.set_h_p()
  def set_h_p(self):
    if self.args.scheduler=="adamW":
      weights_for_dataset={}
      weights_for_dataset["w_dense_loss_data_S"]=20000000.0
      weights_for_dataset["w_dense_loss_data_R"]=20000000.0
      weights_for_dataset["w_dense_loss_data_I"]=20000000.0
      weights_for_dataset["w_dense_loss_grad_S"]=5.0
      weights_for_dataset["w_dense_loss_grad_R"]=5.0
      weights_for_dataset["w_dense_loss_grad_I"]=5.0
      weights_for_dataset["w_cgmit_loss_data_S"]=0.005
      weights_for_dataset["w_cgmit_loss_data_R"]=0.005
      weights_for_dataset["w_cgmit_loss_data_I"]=0.005
      weights_for_dataset["w_cgmit_loss_grad_R"]=5.0
      weights_for_dataset["w_cgmit_loss_grad_S"]=5.0
      weights_for_dataset["w_cgmit_loss_grad_I"]=0.00005
      weights_for_dataset["w_rsmooth_loss_final"]=300000.0
      weights_for_dataset["w_en_ob_iiw_loss"]=0.000005
      weights_for_dataset["w_saw_loss"]=0.05
      weights_for_dataset["w_bigtime_loss_I"]=0.000005
      weights_for_dataset["w_bigtime_loss_r"]=0.5
      weights_for_dataset["wd"]=0
      return weights_for_dataset
    else:
      weights_for_dataset={}
      weights_for_dataset["w_dense_loss_data_S"]=10000000000.0
      weights_for_dataset["w_dense_loss_data_R"]=10000000000.0
      weights_for_dataset["w_dense_loss_data_I"]=10000000000.0
      weights_for_dataset["w_dense_loss_grad_S"]=100000.0
      weights_for_dataset["w_dense_loss_grad_R"]=10000.0
      weights_for_dataset["w_dense_loss_grad_I"]=10000.0
      weights_for_dataset["w_cgmit_loss_data_S"]=1000000000.0
      weights_for_dataset["w_cgmit_loss_data_R"]=1000000000.0
      weights_for_dataset["w_cgmit_loss_data_I"]=1000000000.0
      weights_for_dataset["w_cgmit_loss_grad_R"]=1000.0
      weights_for_dataset["w_cgmit_loss_grad_S"]=1000.0
      weights_for_dataset["w_cgmit_loss_grad_I"]=1000.0
      weights_for_dataset["w_rsmooth_loss_final"]=300000000.0
      weights_for_dataset["w_en_ob_iiw_loss"]=10000.0
      weights_for_dataset["w_saw_loss"]=1.0
      weights_for_dataset["w_bigtime_loss_I"]=0.5
      weights_for_dataset["w_bigtime_loss_r"]=000.1
      weights_for_dataset["wd"]=0.0
      return weights_for_dataset
