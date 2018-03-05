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

le = cPickle.load(open('val/label-encoder.pkl','rb'))
rawdata_root = '/media/gserver/data/tianwen/rawdata'

test_index = pd.read_csv(os.path.join(rawdata_root, 'first_test_index_20180131.csv'))

files = ['resnet20_crop-0.8138-aug8-0.8229.npy',
         'xception-0.8138.npy',
         'xception_crop-0.7995-aug8-0.8129.npy',
         ]



scores = []
for file_name in files:
    file_path = os.path.join('./online_pred/scores',file_name)
    print file_path
    score = np.load(file_path)
    scores.append(score)

scores_pred = np.mean(np.array(scores),axis=0)
test_preds = np.argmax(scores_pred,axis=1)
pred_all = le.inverse_transform(test_preds)

# save csv file to submit
model_name = 'merge-resnet20_crop_aug-xception-xception_crop_aug'
sub = pd.DataFrame({'id':test_index['id'].tolist(),
                    'type':pred_all})

sub[['id','type']].to_csv('./online_pred/subs/%s.csv'%(model_name),index=False,header=None)
print sub
print sub['type'].value_counts()



