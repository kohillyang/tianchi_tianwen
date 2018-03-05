#coding=utf8
from __future__ import division
import torch
import os,time,datetime
from torch.autograd import Variable
import logging
import torch.nn.functional as F
import numpy as np
from math import ceil
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report

def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')



def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def train(model,
          epoch_num,
          batch_size,
          start_epoch,
          optimizer,
          optimizer_center_loss,
          nllloss,
          criterion_center_loss,
          exp_lr_scheduler,
          data_set,
          data_loader,
          usecuda,
          save_inter,
          save_dir,
          augloss=False
          ):

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(start_epoch,epoch_num):
        t_s = time.time()


        # train phase
        exp_lr_scheduler.step(epoch)
        logging.info( 'current lr:%s'%exp_lr_scheduler.get_lr())
        model.train(True)  # Set model to training mode


        for batch_cnt, data in enumerate(data_loader['train']):
            model.train(True)
            # print data
            inputs, labels = data

            if usecuda:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

            else:
                inputs = Variable(inputs)
                labels = Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            optimizer_center_loss.zero_grad()
            # forward
            assert augloss == False
            if augloss:
                outputs,aux = model(inputs)
                loss = criterion(outputs, labels) + criterion(labels, labels)
            else:
                ip1, pred = model(inputs)
                loss = nllloss(pred, labels) + criterion_center_loss(labels,ip1)
#             _, preds = torch.max(F.softmax(outputs), 1)
            _,preds = torch.max(pred,1)
            loss.backward()
            optimizer.step()
            optimizer_center_loss.step()
            batch_corrects = torch.sum((preds == labels)).data[0]
            batch_acc = batch_corrects / (labels.size(0))
            batch_f1 = f1_score(labels.data.cpu().numpy(), preds.data.cpu().numpy(), average='macro')


            # batch loss
            if batch_cnt % 200 == 0:
                unique, counts = np.unique(labels.data.cpu().numpy(), return_counts=True)

                logging.info('%s [%d-%d] | batch-loss: %.3f | f1: %.3f | %s'
                             % (dt(), epoch, batch_cnt, loss.data[0], batch_f1, counts))


            if batch_cnt % 3500 == 0:
                # val phase
                model.train(False)  # Set model to evaluate mode

                val_loss = 0
                val_corrects = 0
                val_size = ceil(len(data_set['val']) / data_loader['val'].batch_size)
                val_preds = np.zeros(len(data_set['val']))+100
                val_true = np.zeros(len(data_set['val']))+100
                t0 = time.time()
                idx = 0


                for batch_cnt_val,data_val in enumerate(data_loader['val']):
                    # print data
                    inputs, labels = data_val

                    if usecuda:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())

                    else:
                        inputs = Variable(inputs)
                        labels = Variable(labels)

                    # forward
                    ip1, pred = model(inputs)
                    loss = nllloss(pred, labels) + criterion_center_loss(labels,ip1)
                    #             _, preds = torch.max(F.softmax(outputs), 1)
                    _,preds = torch.max(pred,1)

                    # statistics
                    val_loss += loss.data[0]
                    batch_corrects = torch.sum((preds == labels)).data[0]

                    val_preds[idx:(idx+labels.size(0))] = preds.data.cpu().numpy()
                    val_true[idx:(idx + labels.size(0))] = labels.data.cpu().numpy()

                    val_corrects += batch_corrects

                    idx += labels.size(0)

                val_loss = val_loss / val_size
                val_f1 = f1_score(val_true, val_preds,average='macro')
                val_report = classification_report(val_true, val_preds, target_names=data_set['val'].classes)
                unique, counts = np.unique(val_preds, return_counts=True)

                t1 = time.time()
                since = t1-t0
                logging.info('--'*30)
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                logging.info('%s'%val_report)
                logging.info('pred unique: %s' % unique)
                logging.info('pred count: %s'%counts)
                logging.info('%s epoch[%d]-val-loss: %.4f ||val-f1@1 : %.4f||time: %d'
                             % (dt(), epoch, val_loss, val_f1, since))
                logging.info('--' * 30)

                if val_f1 > best_acc:
                    best_acc = val_f1
                    best_model_wts = model.state_dict()


                # save model
                save_path = os.path.join(save_dir,
                        'weights-%d-%d-[%.4f].pth'%(epoch,batch_cnt,val_f1))
                torch.save(model.state_dict(), save_path)
                logging.info('saved model to %s' % (save_path))
        t_e = time.time()
        logging.info('----time cost: %d sec'%(t_e-t_s))
        logging.info('===' * 20)


    return best_acc,best_model_wts