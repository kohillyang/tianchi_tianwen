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
from utils.train_center_loss import train,trainlog
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from twdata.twdataset import TWdata
from models.xception_1d_center_loss import xception
from sklearn.model_selection import train_test_split
from models.LeNet import LeNet,MLPNet
import logging
from twdata.twaugment import Compose, AddNoise, RandomAmplitude, DownSample, FlowNormalize, \
                        AddAxis, CenterCrop, RandomShiftCrop
from CenterLoss import CenterLoss
from config import config_edict
class TWAug(object):
    def __init__(self):
        self.augment = Compose([
            AddNoise(A=0.1),
            RandomAmplitude(l=0.9,h=1.1),
            RandomShiftCrop(),
            FlowNormalize(),
            AddAxis()
        ])

    def __call__(self, spct):
        return self.augment(spct)

class TWAugVal(object):
    def __init__(self):
        self.augment = Compose([
            CenterCrop(),
            FlowNormalize(),
            AddAxis()
        ])

    def __call__(self, spct):
        return self.augment(spct)




os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

rawdata_root = '/home/kohill/hszc/data/tianchi/tian_wen_shu_ju_wa_jue'
usecuda = 1
start_epoch = 0
batch_size = 128*3
epoch_num = 130
save_inter = 5



# data prepare
train_data_root = os.path.join(rawdata_root, 'first_train_data')
test_data_root = os.path.join(rawdata_root, 'first_test_data')

train_index = pd.read_csv(os.path.join(rawdata_root, 'first_train_index_20180131.csv'))
test_index = pd.read_csv(os.path.join(rawdata_root, 'first_test_index_20180131.csv'))


le = LabelEncoder()
train_index['type'] = le.fit_transform(train_index['type'])
print le.classes_
fake_label = [0] * test_index.shape[0]
ids = train_index['id'].tolist()


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

print index_val.type.value_counts()


#
#
#
data_set = {}
data_set['train'] = TWdata(index_pd = repeat_index_train,
                           data_root=train_data_root,
                           classes = le.classes_,
                           transform=TWAug(),
                           )

data_set['val'] = TWdata(index_pd = index_val,
                         data_root=train_data_root,
                            classes = le.classes_,
                            transform=TWAugVal(),
                         )
#
data_loader = {}
data_loader['train'] = torchdata.DataLoader(data_set['train'], 32*3, num_workers=4,
                                            shuffle=True, pin_memory=True)
data_loader['val'] = torchdata.DataLoader(data_set['val'], batch_size=128*3, num_workers=4,
                                          shuffle=False, pin_memory=True)

print data_loader['val'].batch_size

print 'dataset: %d,%d'%(len(data_set['train']), len(data_set['val']))



# model prepare
resume = "/home/kohill/Desktop/tianchi/tianwen/output/xception_centerloss/weights-35-10500-[0.8096].pth"
model = xception(num_classes=4)

model = torch.nn.DataParallel(model)
if resume:
    print('resuming finetune from %s'%resume)
    logging.info('resuming finetune from %s'%resume)
    model.load_state_dict(torch.load(resume))

model = model.cuda()



optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
criterion = CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)



save_dir = 'output/xception_centerloss/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = 'output/xception_centerloss/trainlog.log'
trainlog(logfile)
#############something for center loss
nllloss = nn.NLLLoss().cuda()
center_loss = CenterLoss(4,256,1.0).cuda()
optimizer_center_loss = optim.SGD(center_loss.parameters(), lr =0.05)
#############
best_acc,best_model_wts = train(model,
                                epoch_num,
                                batch_size,
                                start_epoch,
                                optimizer,
                                optimizer_center_loss,
                                nllloss,
                                center_loss,
                                exp_lr_scheduler,
                                data_set,
                                data_loader,
                                usecuda,
                                save_inter,
                                save_dir)


save_path = os.path.join(save_dir,'bestweight-[%.4f].pth'%(best_acc))
torch.save(model.state_dict(), save_path)
logging.info('saved model to %s' % (save_path))
