"""
cnn for 1-d signal data, pytorch version
 
Shenda Hong, Jan 2020
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
import os
from imblearn.over_sampling import RandomOverSampler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummaryX import summary

from resnet1d.resnet1d import MyDataset, ResNet1D

def label2index(i):
    m = {'N':0, 'S':1, 'V':2, 'F':3, 'Q':4} # uncomment for 5 classes
    # m = {'N':0, 'S':0, 'V':1, 'F':0, 'Q':0} # uncomment for 2 classes
    return m[i]

if __name__ == "__main__":

    batch_size = 256
    path = 'data/'

    # read data
    print('start')
    data = np.load(os.path.join(path, 'mitdb_data.npy'))
    label_str = np.load(os.path.join(path, 'mitdb_group.npy'))
    label = np.array([label2index(i) for i in label_str])
    
    # make data
    train_ind = np.load(os.path.join(path, 'mitdb_train_ind.npy'))
    test_ind = np.load(os.path.join(path, 'mitdb_test_ind.npy'))
    data = preprocessing.scale(data, axis=1)
    X_train = data[train_ind]
    X_test = data[test_ind]
    Y_train = label[train_ind]
    Y_test = label[test_ind]
    print(X_train.shape, Counter(Y_train))
    print(X_test.shape, Counter(Y_test))
    ros = RandomOverSampler(random_state=0)
    X_train, Y_train = ros.fit_resample(X_train, Y_train)
    print(X_train.shape, Counter(Y_train))
    print(np.max(X_train), np.min(X_train))
    # for i in range(20):
    #     plt.figure()
    #     idx = np.random.randint(X_train.shape[0])
    #     title = '{}_{}'.format(Y_train[idx], idx)
    #     plt.plot(X_train[idx])
    #     plt.title(title)
    #     plt.savefig('img/{0}.png'.format(title))
    # exit()
    
    # prepare loader
    shuffle_idx = np.random.permutation(list(range(X_train.shape[0])))
    X_train = X_train[shuffle_idx]
    Y_train = Y_train[shuffle_idx]
    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)
    dataset = MyDataset(X_train, Y_train)
    dataset_test = MyDataset(X_test, Y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False, shuffle=False)
    
    # make model
    device_str = "cuda:5"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(device)
    model = ResNet1D(
        in_channels=1, 
        base_filters=128, 
        kernel_size=16, 
        stride=2, 
        groups=8, 
        n_block=8, 
        n_classes=len(np.unique(Y_train)), 
        downsample_gap=2, 
        increasefilter_gap=4, 
        use_do=False)
    summary(model, torch.zeros(1, 1, 360))
    model.to(device)

    # train and test
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss()

    n_epoch = 30
    step = 0
    prev_f1 = 0
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):

        # train
        model.train()
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        all_pred_prob_train = []
        for batch_idx, batch in enumerate(prog_iter):
            model.train()
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            all_pred_prob_train.append(pred.cpu().data.numpy())
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            # test
            if batch_idx % 300 == 0:
                print(loss.item())
                model.eval()
                prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
                all_pred_prob = []
                with torch.no_grad():
                    for batch_idx, batch in enumerate(prog_iter_test):
                        input_x, input_y = tuple(t.to(device) for t in batch)
                        pred = model(input_x)
                        all_pred_prob.append(pred.cpu().data.numpy())
                all_pred_prob = np.concatenate(all_pred_prob)
                all_pred = np.argmax(all_pred_prob, axis=1)
                print(classification_report(Y_test, all_pred))
                print(confusion_matrix(Y_test, all_pred))

        scheduler.step(_)
        all_pred_prob_train = np.concatenate(all_pred_prob_train)
        all_pred_train = np.argmax(all_pred_prob_train, axis=1)
        print(classification_report(Y_train, all_pred_train))
        print(confusion_matrix(Y_train, all_pred_train))
        


