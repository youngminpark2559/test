import sys,os,copy,argparse
import traceback

def return_bs_paths(iterator,batch_size):
  try:
    bs_paths=[]
    for i in range(batch_size):
      one_path=next(iterator)
      bs_paths.append(one_path)

    # print("bs_paths",bs_paths)
    # [['/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/kaggle.com_saketc_skin-lesion-analyser-sla/base_dir/train_dir/bcc/_366_1646016.jpg', 'bcc'],
    #  ['/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/kaggle.com_saketc_skin-lesion-analyser-sla/base_dir/train_dir/vasc/_73_3485434.jpg', 'vasc'],
    #  ['/mnt/1T-5e7/Code_projects/Bio_related/Skin_related/kaggle.com_kmader_skin-cancer-mnist-ham10000/kaggle.com_saketc_skin-lesion-analyser-sla/base_dir/train_dir/nv/ISIC_0030157.jpg', 'nv']]

    return bs_paths

  except:
    print(traceback.format_exc())
    print("Error when loading images")
    print("path_to_be_loaded",bs_paths)
