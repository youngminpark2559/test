# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

import Augmentor
import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import traceback

# ================================================================================
def create_img_path_and_class_pairs(paths_image):
  img_and_class_pair=[]
  for one_img_path in paths_image:
    img_path=class_name=one_img_path.replace("\n","")
    class_name=one_img_path.split("/")[-2]
    img_and_class_pair.append([img_path,class_name])

  # print("img_and_class_pair",img_and_class_pair)
  # [['/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/kaggle.com_saketc_skin-lesion-analyser-sla/base_dir/train_dir/akiec/_0_133028.jpg', 'akiec'], 
  #  ['/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/kaggle.com_saketc_skin-lesion-analyser-sla/base_dir/train_dir/akiec/_0_1605139.jpg', 'akiec'], 

  return img_and_class_pair