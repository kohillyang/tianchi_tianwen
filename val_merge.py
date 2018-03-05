#coding=utf8
from __future__ import division
from torch import nn
import torch
import torch.utils.data as torchdata
from torchvision import datasets,transforms
import os,time
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from glob import glob
import cPickle

val_true = np.load('val/val-labels.npy')
le = cPickle.load(open('val/label-encoder.pkl','rb'))


files = ['resnet20_crop_aug8-0.8229.npy',
         'xception-0.8138.npy',
         'xception_crop-0.7995-aug8-0.8129.npy',
         ]



scores = []
for file_name in files:
    file_path = os.path.join('./val',file_name)
    print file_path
    score = np.load(file_path)
    scores.append(score)

scores_pred = np.mean(np.array(scores),axis=0)
val_preds = np.argmax(scores_pred,axis=1)
val_f1 = f1_score(val_true, val_preds,average='macro')
print val_f1


