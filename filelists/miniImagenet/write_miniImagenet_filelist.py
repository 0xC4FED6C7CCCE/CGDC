import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re

cwd = os.getcwd()
# data_path = join(cwd,'ILSVRC2015/Data/CLS-LOC/train')
data_path = join('/home/dv/Desktop/Few_Shot_Distribution_Calibration-master/Datasets/miniImagenet/data')
savedir = './'
dataset_list = ['base', 'val', 'novel']

#if not os.path.exists(savedir):
#    os.makedirs(savedir)

cl = -1
folderlist = []

datasetmap = {'base':'train','val':'val','novel':'test'};
filelists = {'base':{},'val':{},'novel':{} }
filelists_flat = {'base':[],'val':[],'novel':[] }
labellists_flat = {'base':[],'val':[],'novel':[] }

for dataset in dataset_list:
    with open(datasetmap[dataset] + ".csv", "r") as lines:
        for i, line in enumerate(lines):
            if i == 0:
                continue
            fid, _ , label = re.split(',|\.', line)
            # print(fid)
            # print(label)
            label = label.replace('\n','')
            if not label in filelists[dataset]:
                folderlist.append(label)
                filelists[dataset][label] = []
                fnames = listdir(join(data_path, label))
                # print(fnames)
                # print(re.split(',|\.', re.split('00000', fnames[0])[1])[0])
                # print(fnames[0][-8:-4])
                fname_number = [int(fname[-8:-4]) for fname in fnames]
                sorted_fnames = list(zip( *sorted(  zip(fnames, fname_number), key = lambda f_tuple: f_tuple[1] )))[0]
                print(sorted_fnames)
            # print(sorted_fnames)
            # print(fid)
            fid = int(fid[-5:])-1
            # print(fid)
            # print(len(sorted_fnames))
            for i in sorted_fnames:
                fname = join( data_path,label, i)
                # print(fname)
                filelists[dataset][label].append(fname)

    for key, filelist in filelists[dataset].items():
        cl += 1
        random.shuffle(filelist)
        filelists_flat[dataset] += filelist
        labellists_flat[dataset] += np.repeat(cl, len(filelist)).tolist() 

for dataset in dataset_list:
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folderlist])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in filelists_flat[dataset]])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in labellists_flat[dataset]])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
