import re
import os
from shutil import copyfile

# path = './data/datasets/tiny-imagenet-200/images/val'
path = './data/datasets/tiny-imagenet-200/val'

path_images = path + '/' + 'images'

path_file = path + '/' + 'val_annotations.txt'

with open(path_file) as f: classes = f.readlines()


for i in range(len(classes)):
    print (i)
    # import pdb
    # pdb.set_trace()
    this_str = re.split(r'\t+', classes[i])
    this_file_name = this_str[0]
    this_class_name = this_str[1]

    ## make dir
    this_path_name = path + '/' + this_class_name
    if not os.path.exists(this_path_name):
        os.makedirs(this_path_name)
    #### copy file
    this_file_path = path_images + '/' + this_file_name
    this_target_file_path = this_path_name + '/' + this_file_name
    copyfile(this_file_path, this_target_file_path)




