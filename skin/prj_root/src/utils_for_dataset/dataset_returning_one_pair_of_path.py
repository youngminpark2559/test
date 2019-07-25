
# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/utils_for_dataset/dataset_cgmit.py
# /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/prj_root/src/unit_test/Test_dataset_cgmit.py

# ================================================================================
from random import shuffle

# ================================================================================
import torch.utils.data as data

# ================================================================================
from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image
from src.utils import utils_data_process as utils_data_process

# ================================================================================
class Dataset_Returning_One_Pair_Of_Path(data.Dataset):
  def __init__(self,path_of_images,args):
    self.args=args

    # ================================================================================
    paths_image,self.num_of_imgs=utils_common.return_path_list_from_txt(path_of_images)

    # ================================================================================
    img_and_class_pair=utils_data_process.create_img_path_and_class_pairs(paths_image=paths_image)

    # ================================================================================
    self.zipped_one=img_and_class_pair

    shuffle(self.zipped_one)

  def __len__(self):
    return self.num_of_imgs

  def __getitem__(self,idx):
    one_pair=self.zipped_one[idx]
    return one_pair
