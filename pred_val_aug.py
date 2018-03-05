#coding=utf8
from __future__ import division
from torch import nn
import torch
import torch.utils.data as torchdata
from torchvision import datasets,transforms
import os,time
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils.train import train,trainlog
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from twdata.twdataset import TWdata
from models.Resnet_1d import resnet20_1d
from models.xception_1d import xception
from models.inceptionV3_1d import Inception3
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from twdata.twaugment import Compose, AddNoise, RandomAmplitude, DownSample, FlowNormalize, \
                        AddAxis, ShiftCrop
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report


class TWAugVal(object):
    def __init__(self, start_point,end_point):
        self.augment = Compose([
            ShiftCrop(start_point=start_point,end_point=end_point),
            FlowNormalize(),
            AddAxis()
        ])
    def __call__(self, spct):
        return self.augment(spct)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

rawdata_root = '/media/gserver/data/tianwen/rawdata'
usecuda = 1
bs = 128*3


# data prepare
train_data_root = os.path.join(rawdata_root, 'first_train_data')

train_index = pd.read_csv(os.path.join(rawdata_root, 'first_train_index_20180131.csv'))
print train_index.type.value_counts()

le = LabelEncoder()
train_index['type'] = le.fit_transform(train_index['type'])
print le.classes_


index_train, index_val= train_test_split(train_index,
                                         test_size=0.1, random_state=42,
                                         stratify=train_index['type'])




# model prepare
resume = '/media/gserver/models/tianwen/resnet20_crop_noi2/weights-53-14000-[0.8130].pth'
model = resnet20_1d(num_classes=4)


model = torch.nn.DataParallel(model)
if resume:
    print('resuming finetune from %s'%resume)
    model.load_state_dict(torch.load(resume))

model = model.cuda()

model.train(False)  # Set model to evaluate mode

data_set = {}
data_set['val'] = TWdata(index_pd=index_val,
                         data_root=train_data_root,
                         classes=le.classes_,
                         )



num_crop = 8

val_true = np.zeros(len(data_set['val']),dtype=int)

scores_pred = np.zeros((num_crop,len(data_set['val']),4),dtype=np.float32)
q = np.linspace(0, 2600-int(2600*0.8), num_crop,dtype=int)

for aug_cnt, crop_start in enumerate(q):
    crop_end = crop_start+int(2600*0.8)
    data_set = {}

    data_set['val'] = TWdata(index_pd=index_val,
                             data_root=train_data_root,
                             classes=le.classes_,
                             transform=TWAugVal(start_point=crop_start,end_point=crop_end),
                             )

    data_loader = {}
    data_loader['val'] = torchdata.DataLoader(data_set['val'], batch_size=bs, num_workers=4,
                                              shuffle=False, pin_memory=True)


    idx = 0
    for batch_cnt, data in enumerate(data_loader['val']):

        print 'aug %d, %d/%d'%(aug_cnt,batch_cnt,len(data_set['val'])//bs)
        inputs, labels = data

        if usecuda:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        # forward
        outputs = model(inputs)
        # print outputs.size()

        score = F.softmax(outputs)

        scores_pred[aug_cnt,idx:idx+score.size(0),:] = score.data.cpu().numpy()
        val_true[idx:idx+score.size(0)] = labels.numpy()
        idx = idx + score.size(0)

scores_pred = scores_pred.mean(axis=0)
val_preds = np.argmax(scores_pred,axis=1)
val_f1 = f1_score(val_true, val_preds,average='macro')
print val_f1

model_name = 'resnet20_crop_noi2-0.8030-aug8'
np.save('./val/%s-%.4f.npy'%(model_name, val_f1),scores_pred)


# np.save('./val/val-labels.npy',val_true)
#
# import cPickle
# cPickle.dump(le,open('./val/label-encoder.pkl','wb'))



