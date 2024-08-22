import os
import numpy as np


os.makedirs('/Users/maahirpaliwal/Downloads/archive/imgs_zip/output')
os.makedirs('/Users/maahirpaliwal/Downloads/archive/imgs_zip/output/train')
os.makedirs('/Users/maahirpaliwal/Downloads/archive/imgs_zip/output/val')
os.makedirs('/Users/maahirpaliwal/Downloads/archive/imgs_zip/output/test')

os.listdir('/Users/maahirpaliwal/Downloads/archive/imgs_zip/output')

import shutil
import random
import math

root_dir = '/Users/maahirpaliwal/Downloads/archive/imgs_zip/imgs'
classes = ['Acura', 'Alfa Romeo', 'Aston Martin', 'Audi', 'Bentley', 'BMW']

for clss in classes:
    print('------------' + clss + '-------------')
    dirtry = root_dir + '/' + clss
    files = os.listdir(dirtry)
    np.random.shuffle(files)        


    base_outdir = '/Users/maahirpaliwal/Downloads/archive/imgs_zip/output/'

    for folder in ['train', 'val', 'test']:
        target_dir = base_outdir + folder
        os.makedirs(target_dir + '/' + clss)
        target_class = target_dir + '/' + clss

        if folder == 'train':
            images_to_pass = files[: math.floor(0.8*len(files))]
            for img in images_to_pass:
                img = dirtry + '/' + img
                shutil.copy(img, target_class)
        elif folder == 'val':
            images_to_pass = files[math.floor(0.8*len(files)): math.floor(0.9*len(files))]
            for img in images_to_pass:
                img = dirtry + '/' + img
                shutil.copy(img, target_class)
        else:
            images_to_pass = files[math.floor(0.9*len(files)):]
            for img in images_to_pass:
                img = dirtry + '/' + img
                shutil.copy(img, target_class)