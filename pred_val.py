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
                        AddAxis,CenterCrop
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report


class TWAugVal(object):
    def __init__(self):
        self.augment = Compose([
            FlowNormalize(),
            AddAxis()
        ])
    def __call__(self, spct):
        return self.augment(spct)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

rawdata_root = '/media/gserver/data/tianwen/rawdata'
usecuda = 1


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



class_sample_count = index_train.type.value_counts()
weights =  [int(class_sample_count.max()/class_sample_count[x]) for x in range(len(class_sample_count))]
print weights

repeat_index_train = pd.DataFrame()
for i,weight in enumerate(weights):
    temp = index_train[index_train.type == i]
    temp = pd.concat([temp] * weight, ignore_index=True)
    repeat_index_train = pd.concat([repeat_index_train, temp], axis=0, ignore_index=True)


print repeat_index_train.type.value_counts()



data_set = {}

data_set['val'] = TWdata(index_pd = index_val,
                         data_root=train_data_root,
                            classes = le.classes_,
                            transform=TWAugVal(),
                         )


data_loader = {}
data_loader['val'] = torchdata.DataLoader(data_set['val'], batch_size=128*3, num_workers=4,
                                          shuffle=False, pin_memory=True)


print 'dataset: %d'%len(data_set['val'])



# model prepare
resume = '/media/gserver/models/tianwen/xception/weights-29-9000-[0.8138].pth'
# model = resnet34_1d(num_classes=4)
model = xception(num_classes=4)

model = torch.nn.DataParallel(model)
if resume:
    print('resuming finetune from %s'%resume)
    model.load_state_dict(torch.load(resume))

model = model.cuda()

model.train(False)  # Set model to evaluate mode

scores_pred = np.zeros((len(data_set['val']),4),dtype=np.float32)
val_true = np.zeros(len(data_set['val']),dtype=int)


idx = 0
for batch_cnt, data in enumerate(data_loader['val']):

    # print data
    inputs, labels = data

    if usecuda:
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)

    # forward
    outputs = model(inputs)
    # print outputs.size()

    score = F.softmax(outputs)

    scores_pred[idx:idx+score.size(0),:] = score.data.cpu().numpy()
    val_true[idx:idx+score.size(0)] = labels.numpy()
    idx = idx + score.size(0)

val_preds = np.argmax(scores_pred,axis=1)
val_f1 = f1_score(val_true, val_preds,average='macro')
print val_f1

model_name = 'xception'
np.save('./val/%s-%.4f.npy'%(model_name, val_f1),scores_pred)
np.save('./val/val-labels.npy',val_true)

import cPickle
cPickle.dump(le,open('./val/label-encoder.pkl','wb'))
