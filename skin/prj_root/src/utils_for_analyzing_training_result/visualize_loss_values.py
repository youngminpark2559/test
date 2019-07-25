import sys,os,copy,argparse

class Visualize_Loss_Values():
  def __init__(self):
    self.loss_r_data_dense=[]
    self.loss_r_grad_dense=[]
    self.loss_s_data_dense=[]
    self.loss_s_grad_dense=[]
    self.loss_o_data_dense=[]
    self.loss_o_grad_dense=[]
    self.iiw_loss_temp=[]
    self.saw_loss_temp=[]
    self.bigtime_loss_temp=[]
    self.bigtime_loss_I_temp=[]
    self.bigtime_loss_I_seq_temp=[]
    self.bigtime_loss_r_temp=[]

    self.lo_list=[
      ("loss_r_data_dense",self.loss_r_data_dense),
      ("loss_r_grad_dense",self.loss_r_grad_dense),
      ("loss_s_data_dense",self.loss_s_data_dense),
      ("loss_s_grad_dense",self.loss_s_grad_dense),
      ("loss_o_data_dense",self.loss_o_data_dense),
      ("loss_o_grad_dense",self.loss_o_grad_dense),
      ("iiw_loss_temp",self.iiw_loss_temp),
      ("saw_loss_temp",self.saw_loss_temp),
      ("bigtime_loss_temp",self.bigtime_loss_temp),
      ("bigtime_loss_I_temp",self.bigtime_loss_I_temp),
      ("bigtime_loss_I_seq_temp",self.bigtime_loss_I_seq_temp),
      ("bigtime_loss_r_temp",self.bigtime_loss_r_temp),]
    self.num_lo=len(self.lo_list)
