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
from models.Resnet_1d import resnet34_1d,resnet20_1d
from models.xception_1d import xception
from models.xception_1d_center_loss import xception

from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from twdata.twaugment import Compose, AddNoise, RandomAmplitude, DownSample, FlowNormalize, \
                        AddAxis, ShiftCrop

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

rawdata_root = '/home/kohill/hszc/data/tianchi/tian_wen_shu_ju_wa_jue'
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




data_set = {}

data_set['test'] = TWdata(index_pd = test_index,
                         data_root=test_data_root,
                            classes = le.classes_,
                         )


# model prepare
resume = 'output/xception_centerloss_lr/weights-4-7000-[0.8215].pth'
model = xception(num_classes=4)

model = torch.nn.DataParallel(model)
if resume:
    print('resuming finetune from %s'%resume)
    model.load_state_dict(torch.load(resume))

model = model.cuda()
model.train(False)  # Set model to evaluate mode


num_crop=8

scores_pred = np.zeros((num_crop, len(data_set['test']),4),dtype=np.float32)
q = np.linspace(0, 2600-int(2600*0.8), num_crop,dtype=int)
for aug_cnt, crop_start in enumerate(q):
    crop_end = crop_start+int(2600*0.8)
    data_set = {}

    data_set['test'] = TWdata(index_pd=test_index,
                             data_root=test_data_root,
                             classes=le.classes_,
                             transform=TWAugVal(start_point=crop_start,end_point=crop_end),
                             )

    data_loader = {}
    data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=bs, num_workers=4,
                                              shuffle=False, pin_memory=True)
    idx = 0
    for batch_cnt, data in enumerate(data_loader['test']):

        print 'aug %d, %d/%d' % (aug_cnt, batch_cnt, len(data_set['test']) // bs)
        inputs, labels = data

        if usecuda:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        # forward
        gb,outputs = model(inputs)
#         score = F.softmax(outputs)    
        score = outputs
        scores_pred[aug_cnt, idx:idx+score.size(0),:] = score.data.cpu().numpy()

        idx = idx + score.size(0)

scores_pred = scores_pred.mean(axis=0)
test_preds = np.argmax(scores_pred,axis=1)
pred_all = le.inverse_transform(test_preds)


model_name = 'weights-4-7000-[0.8215]'
# save score
print scores_pred.shape
np.save('./online_pred/scores/%s.npy'%(model_name),scores_pred)



# save csv file to submit
sub = pd.DataFrame({'id':test_index['id'].tolist(),
                    'type':pred_all})

sub[['id','type']].to_csv('./online_pred/subs/%s.csv'%(model_name),index=False,header=None)
from pprint import pprint
pprint(sub)
print sub['type'].value_counts()

