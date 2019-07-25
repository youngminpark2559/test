import sys,os,copy,argparse

class Path_Of_Text_Files():
  def __init__(self,args):
    self.cgintrinsic_trn=args.text_file_for_paths_dir+"/cgintrinsic_trn.txt"
    self.cgintrinsic_rgt=args.text_file_for_paths_dir+"/cgintrinsic_rgt.txt"
    self.iiw_trn=args.text_file_for_paths_dir+"/iiw_trn.txt"
    self.iiw_json=args.text_file_for_paths_dir+"/iiw_json.txt"
    self.saw_trn=args.text_file_for_paths_dir+"/saw_trn.txt"
    self.saw_npy=args.text_file_for_paths_dir+"/saw_npy.txt"
    self.bigtime_trn=args.text_file_for_paths_dir+"/bigtime_trn.txt"
    self.bigtime_mask=args.text_file_for_paths_dir+"/bigtime_mask.txt"
