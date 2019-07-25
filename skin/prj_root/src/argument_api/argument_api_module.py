import datetime
import sys,os,copy,argparse

class Argument_API_class(argparse.ArgumentParser):
  def __init__(self):
    super(Argument_API_class,self).__init__()

    self.add_argument("--train_mode",default=True)
    self.add_argument("--batch_size",help=2)
    self.add_argument("--epoch",help=2)
    self.add_argument("--use_saved_model",default=False)
    self.add_argument("--model_save_dir",help="./ckpt")
    self.add_argument("--model_file_name_when_saving_and_loading_model",help="/custom_IID.pth")
    self.add_argument("--use_local_piecewise_constant_loss",default=False)
    self.add_argument("--use_augmentor",default=False)
    self.add_argument("--use_integrated_decoders",default=True)
    self.add_argument("--use_loss_display",default=True)
    self.add_argument("--input_size")
    self.add_argument("--leaping_batchsize_for_saving_model",help=\
      "For example, if it's 1, you'll save model every batch\n\
      if leaping_batchsize_for_saving_model is 100 with 4 batch size,\n\
      you'll save model every 100*4*1=400, 100*4*2=800, ... batch \n\
      ...\n\
      Integer should be > 0")
    self.add_argument("--use_multi_gpu",default=False)
    self.add_argument("--text_file_for_paths_dir")
    self.add_argument("--iiw_training_strategy")
    self.add_argument("--check_input_output_via_multi_gpus",default=False)
    self.add_argument("--measure_train_time",default=False)
    self.add_argument("--scheduler",default=None)
    self.add_argument("--train_method",help=\
      "full,iiw_and_tinghuiz,separately_update_over_full_dataset,\
      separately_update_over_full_dataset_using_custom_dataset_class")
    self.add_argument("--use_visdom",default=False)
    self.add_argument("--save_imgs",default=False)
    self.add_argument("--visualize_loss_values",default=False)
    self.add_argument("--use_focal_loss",default=False)

    # ================================================================================
    self.args=self.parse_args()
