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
from models.Resnet_1d import resnet34_1d
from models.xception_1d import xception
from models.inceptionV3_1d import Inception3
from models.Resnext_1d import resnext20_type1
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from twdata.twaugment import Compose, AddNoise, RandomAmplitude, DownSample, FlowNormalize, \
                        AddAxis

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
bs = 128*3




# data prepare
train_data_root = os.path.join(rawdata_root, 'first_train_data')
test_data_root = os.path.join(rawdata_root, 'first_test_data')

train_index = pd.read_csv(os.path.join(rawdata_root, 'first_train_index_20180131.csv'))
test_index = pd.read_csv(os.path.join(rawdata_root, 'first_test_index_20180131.csv'))
print train_index.type.value_counts()

le = LabelEncoder()
train_index['type'] = le.fit_transform(train_index['type'])
test_index['type'] = 0
print le.classes_


class_sample_count = train_index.type.value_counts()
weights =  [int(class_sample_count.max()/class_sample_count[x]) for x in range(len(class_sample_count))]
print weights

repeat_index_train = pd.DataFrame()
for i,weight in enumerate(weights):
    temp = train_index[train_index.type == i]
    temp = pd.concat([temp] * weight, ignore_index=True)
    repeat_index_train = pd.concat([repeat_index_train, temp], axis=0, ignore_index=True)


print repeat_index_train.type.value_counts()



data_set = {}

data_set['test'] = TWdata(index_pd = test_index,
                         data_root=test_data_root,
                            classes = le.classes_,
                            transform=TWAugVal(),
                         )


data_loader = {}
data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=bs, num_workers=4,
                                          shuffle=False, pin_memory=True)


print 'dataset: %d'%len(data_set['test'])



# model prepare
resume = '/media/gserver/models/tianwen/xception/weights-19-8000-[0.8100].pth'

model = xception(num_classes=4)

model = torch.nn.DataParallel(model)
if resume:
    print('resuming finetune from %s'%resume)
    model.load_state_dict(torch.load(resume))

model = model.cuda()

model.train(False)  # Set model to evaluate mode


scores_pred = np.zeros((len(data_set['test']),4),dtype=np.float32)

idx = 0
for batch_cnt, data in enumerate(data_loader['test']):

    print '%d/%d' % (batch_cnt, len(data_set['test']) // bs)
    inputs, labels = data

    if usecuda:
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)

    # forward
    outputs = model(inputs)
    score = F.softmax(outputs)

    scores_pred[idx:idx+score.size(0),:] = score.data.cpu().numpy()

    idx = idx + score.size(0)

test_preds = np.argmax(scores_pred,axis=1)
pred_all = le.inverse_transform(test_preds)


model_name = 'xception-0.8100'
# save score
print scores_pred.shape
np.save('./online_pred/scores/%s.npy'%(model_name),scores_pred)



# save csv file to submit
sub = pd.DataFrame({'id':test_index['id'].tolist(),
                    'type':pred_all})

sub[['id','type']].to_csv('./online_pred/subs/%s.csv'%(model_name),index=False,header=None)
print sub
print sub['type'].value_counts()

