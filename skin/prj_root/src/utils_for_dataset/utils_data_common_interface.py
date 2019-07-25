
from src.utils import utils_common as utils_common
from src.utils import utils_data as utils_data
from src.utils import utils_image as utils_image


def load_skin_cancer_dense(bs_paths_dense,args):
  # print("bs_paths_dense",bs_paths_dense)
  # [['/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/kaggle.com_saketc_skin-lesion-analyser-sla/base_dir/train_dir/bcc/_36_6146895.jpg', 'bcc'], ['/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/kaggle.com_saketc_skin-lesion-analyser-sla/base_dir/train_dir/nv/ISIC_0024755.jpg', 'nv'], ['/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/kaggle.com_saketc_skin-lesion-analyser-sla/base_dir/train_dir/bcc/_4_7667536.jpg', 'bcc']]

  # ================================================================================
  lo_img_and_class_data=[]
  for one_img_and_class_path in bs_paths_dense:
    img_path=one_img_and_class_path[0]
    class_name=one_img_and_class_path[1]

    loaded_img=utils_image.load_img(img_path,gray=False)
    print("loaded_img",loaded_img.shape)
    # (450, 600, 3)

    # ================================================================================
    lo_img_and_class_data.append([loaded_img,class_name])
  print("lo_img_and_class_data",lo_img_and_class_data)
  afaf
