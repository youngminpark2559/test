# conda activate py36gputorch041
# cd /mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils/
# rm e.l && python utils_image.py 2>&1 | tee -a e.l && code e.l

import Augmentor
import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import traceback
import os,sys
import glob,natsort
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import utils_common as utils_common
from src.utils import utils_image as utils_image

# ================================================================================
# currentdir = "/mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/utils"
# network_dir="/mnt/1T-5e7/papers/cv/IID/Deep_Adversial_Residual_Network_for_IID/a_c_final/networks"
# sys.path.insert(0,network_dir)

# ================================================================================
def create_gt_df(data_dir,imageid_path_dict,lesion_type_dict):
  df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
  df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
  df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
  df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes
  return df_original

def prepare_dataset(args):
  # utils_common.see_directory(path_of_dir="/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/input")
  
  data_dir = '/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/input'

  # ================================================================================
  all_image_path = glob.glob(os.path.join(data_dir, '*', '*.jpg'))
  # print("all_image_path",all_image_path)
  # ['/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/input/HAM10000_images_part_1/ISIC_0026611.jpg', 
  #  '/mnt/1T-5e7/Code_projects/Bio_related/
  
  # ================================================================================
  imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
  # print("imageid_path_dict",imageid_path_dict)
  # {'ISIC_0026611': '/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/input/HAM10000_images_part_1/ISIC_0026611.jpg',

  # ================================================================================
  lesion_type_dict = {
      'nv': 'Melanocytic nevi',
      'mel': 'dermatofibroma',
      'bkl': 'Benign keratosis-like lesions ',
      'bcc': 'Basal cell carcinoma',
      'akiec': 'Actinic keratoses',
      'vasc': 'Vascular lesions',
      'df': 'Dermatofibroma'
  }

  # ================================================================================
  # norm_mean,norm_std=utils_image.compute_img_mean_std(all_image_path)

  # ================================================================================
  df_original=create_gt_df(data_dir,imageid_path_dict,lesion_type_dict)
  # print("df_original",df_original)
  #          lesion_id      ...      cell_type_idx
  # 0      HAM_0000118      ...                  2
  # 1      HAM_0000118      ...                  2
  # 2      HAM_0002730      ...                  2
  # 3      HAM_0002730      ...                  2

  # ================================================================================
  df_undup = df_original.groupby('lesion_id').count()

  # ================================================================================
  # now we filter out lesion_id's that have only one image associated with it
  df_undup = df_undup[df_undup['image_id'] == 1]
  df_undup.reset_index(inplace=True)

  # ================================================================================
  # print("df_undup.head()",df_undup.head())
  #      lesion_id  image_id  dx  dx_type  age  sex  localization  path  \
  # 0  HAM_0000001  1         1   1        1    1    1             1      
  # 1  HAM_0000003  1         1   1        1    1    1             1      
  # 2  HAM_0000004  1         1   1        1    1    1             1      
  # 3  HAM_0000007  1         1   1        1    1    1             1      
  # 4  HAM_0000008  1         1   1        1    1    1             1      

  #    cell_type  cell_type_idx  
  # 0  1          1              
  # 1  1          1              
  # 2  1          1              
  # 3  1          1              
  # 4  1          1              

  # ================================================================================
  # here we identify lesion_id's that have duplicate images and those that have only one image.

  def get_duplicates(x):
      unique_list = list(df_undup['lesion_id'])
      if x in unique_list:
          return 'unduplicated'
      else:
          return 'duplicated'

  # create a new colum that is a copy of the lesion_id column
  df_original['duplicates'] = df_original['lesion_id']
  # apply the function to this new column
  df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

  # ================================================================================
  # print("df_original.head()",df_original.head())
  #      lesion_id      image_id   dx dx_type   age   sex localization  \
  # 0  HAM_0000118  ISIC_0027419  bkl  histo   80.0  male  scalp         
  # 1  HAM_0000118  ISIC_0025030  bkl  histo   80.0  male  scalp         
  # 2  HAM_0002730  ISIC_0026769  bkl  histo   80.0  male  scalp         
  # 3  HAM_0002730  ISIC_0025661  bkl  histo   80.0  male  scalp         
  # 4  HAM_0001466  ISIC_0031633  bkl  histo   75.0  male  ear           

  #                                                path  \
  # 0  ../input/HAM10000_images_part_1/ISIC_0027419.jpg   
  # 1  ../input/HAM10000_images_part_1/ISIC_0025030.jpg   
  # 2  ../input/HAM10000_images_part_1/ISIC_0026769.jpg   
  # 3  ../input/HAM10000_images_part_1/ISIC_0025661.jpg   
  # 4  ../input/HAM10000_images_part_2/ISIC_0031633.jpg   

  #                         cell_type  cell_type_idx  duplicates  
  # 0  Benign keratosis-like lesions   2              duplicated  
  # 1  Benign keratosis-like lesions   2              duplicated  
  # 2  Benign keratosis-like lesions   2              duplicated  
  # 3  Benign keratosis-like lesions   2              duplicated  
  # 4  Benign keratosis-like lesions   2              duplicated  

  # ================================================================================
  # print("df_original['duplicates'].value_counts()",df_original['duplicates'].value_counts())
  # unduplicated    5514
  # duplicated      4501

  # ================================================================================
  # now we filter out images that don't have duplicates
  df_undup = df_original[df_original['duplicates'] == 'unduplicated']
  # print("df_undup.shape",df_undup.shape)
  # (5514, 11)

  # ================================================================================
  # now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set
  y = df_undup['cell_type_idx']
  _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)
  # print("df_val.shape",df_val.shape)
  # df_val.shape (1103, 11)

  # ================================================================================
  # print("df_val['cell_type_idx'].value_counts()",df_val['cell_type_idx'].value_counts())
  # 4    883
  # 2    88 
  # 6    46 
  # 1    35 
  # 0    30 
  # 5    13 
  # 3    8  

  # ================================================================================
  # This set will be df_original excluding all rows that are in the val set
  # This function identifies if an image is part of the train or val set.
  def get_val_rows(x):
      # create a list of all the lesion_id's in the val set
      val_list = list(df_val['image_id'])
      if str(x) in val_list:
          return 'val'
      else:
          return 'train'

  # ================================================================================
  # identify train and val rows
  # create a new colum that is a copy of the image_id column
  df_original['train_or_val'] = df_original['image_id']

  # apply the function to this new column
  df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)

  # filter out train rows
  df_train = df_original[df_original['train_or_val'] == 'train']
  # print(len(df_train))
  # 8912

  # print(len(df_val))
  # 1103

  # ================================================================================
  # print("df_train['cell_type_idx'].value_counts()",df_train['cell_type_idx'].value_counts())
  # 4    5822
  # 6    1067
  # 2    1011
  # 1    479 
  # 0    297 
  # 5    129 
  # 3    107 

  # ================================================================================
  # print("df_val['cell_type'].value_counts()",df_val['cell_type'].value_counts())
  # Melanocytic nevi                  883
  # Benign keratosis-like lesions     88 
  # dermatofibroma                    46 
  # Basal cell carcinoma              35 
  # Actinic keratoses                 30 
  # Vascular lesions                  13 
  # Dermatofibroma                    8  

  # ================================================================================
  # From From the above statistics of each category, we can see that there is a serious class imbalance in the training data. To solve this problem, I think we can start from two aspects, one is equalization sampling, and the other is a loss function that can be used to mitigate category imbalance during training, such as focal loss.

  # Copy fewer class to balance the number of 7 classes

  # ================================================================================
  # print("df_train",df_train.shape)
  # (8912, 12)

  # print("df_train['cell_type'].value_counts()",df_train['cell_type'].value_counts())
  # Melanocytic nevi                  5822
  # dermatofibroma                    1067
  # Benign keratosis-like lesions     1011
  # Basal cell carcinoma               479
  # Actinic keratoses                  297
  # Vascular lesions                   129
  # Dermatofibroma                     107

  # ================================================================================
  if args.use_focal_loss=="False":
    data_aug_rate = [15,10,5,50,0,40,5]

    # ================================================================================
    for i in range(7):
        if data_aug_rate[i]:
            df_train=df_train.append([df_train.loc[df_train['cell_type_idx'] == i,:]]*(data_aug_rate[i]-1), ignore_index=True)

  # ================================================================================
  # print("df_train",df_train.shape)
  # (35967, 12)

  # print("df_train['cell_type'].value_counts()",df_train['cell_type'].value_counts())
  # Melanocytic nevi                  5822
  # Dermatofibroma                    5350
  # dermatofibroma                    5335
  # Vascular lesions                  5160
  # Benign keratosis-like lesions     5055
  # Basal cell carcinoma              4790
  # Actinic keratoses                 4455

  # ================================================================================
  # At the beginning, I divided the data into three parts, training set, validation set and test set. Considering the small amount of data, I did not further divide the validation set data in practice.

  # We can split the test set again in a validation set and a true test set:
  df_val, df_test = train_test_split(df_val, test_size=0.5)

  # ================================================================================
  df_train = df_train.reset_index()
  df_val = df_val.reset_index()
  df_test = df_test.reset_index()
  
  # ================================================================================
  # print("df_train",df_train.shape)
  # (35967, 13)
  # print("df_val",df_val.shape)
  # (551, 12)
  # print("df_test",df_test.shape)
  # (552, 12)
  
  # ================================================================================
  df_train=df_train.iloc[:5000,:]

  return df_train,df_val,df_test
