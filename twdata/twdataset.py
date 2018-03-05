# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


class TWdata(data.Dataset):
    def __init__(self, index_pd, data_root, classes,transform=None):

        self.ids = index_pd['id'].tolist()
        self.labels = index_pd['type'].tolist()
        self.data_root = data_root
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        label = self.labels[item]
        data_path = os.path.join(self.data_root, '%s.txt'%id)

        spct = np.loadtxt(data_path,delimiter=',')  #(2600, )

        if self.transform is not None:
            spct = self.transform(spct) # (C,2600)


        return torch.from_numpy(spct).float(), label




if __name__ == '__main__':
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from twaugment import TWAug
    import cPickle

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    rawdata_root = '/media/gserver/data/tianwen/rawdata'
    usecuda = 1
    start_epoch = 0
    batch_size = 128 * 3
    epoch_num = 100
    save_inter = 10

    # data prepare
    train_data_root = os.path.join(rawdata_root, 'first_train_data')
    test_data_root = os.path.join(rawdata_root, 'first_test_data')

    train_index = pd.read_csv(os.path.join(rawdata_root, 'first_train_index_20180131.csv'))
    test_index = pd.read_csv(os.path.join(rawdata_root, 'first_test_index_20180131.csv'))

    le = LabelEncoder()
    train_index['type'] = le.fit_transform(train_index['type'])
    fake_label = [0] * test_index.shape[0]
    ids = train_index['id'].tolist()


    data_set = {}
    data_set['train'] = TWdata(index_pd=train_index,
                               data_root=train_data_root,
                               classes=le.classes_,
                               transform=TWAug(NA=0.2, Al=0.9, Ah=1.1),
                               )

    data_loader = {}
    data_loader['train'] = data.DataLoader(data_set['train'], 1024 * 3, num_workers=4,
                                                shuffle=False, pin_memory=True)

    print len(data_set['train'])

    all_data = np.zeros((len(data_set['train']), 2600),dtype=float)
    all_label = np.zeros(len(data_set['train']),dtype=np.int)

    idx=0
    for batch_cnt, batch_data in enumerate(data_loader['train']):
        inputs, labels = batch_data

        inputs = inputs.numpy()[:,0,:]
        labels = labels.numpy()

        all_data[idx : idx + inputs.shape[0]] = inputs
        all_label[idx : idx + inputs.shape[0]] = labels

        idx = idx + inputs.shape[0]


    np.save('all_train_normalized.npy',all_data)
    np.save('all_train_label.npy', all_label)

    # pca = PCA(n_components=30, svd_solver='auto')
    # pca.fit(all_data)
    # print(pca.explained_variance_ratio_)
    # cPickle.dump(pca,open('pca30.pkl','wb'))
    #
