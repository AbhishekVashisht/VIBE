import os
import subprocess
from subprocess izsmport call
all_vid_path=[]
Root_dir="./mp4"
Root_classes=os.listdir(Root_dir)
for vid in Root_classes:
    vid_path=os.path.join(os.path.join(Root_dir,vid))
    call(['python', 'demo.py', '--vid_file=' + vid_path, '--output_folder=opp','--tracking_method=pose','--staf_dir=../openpose'])
