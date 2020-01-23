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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummaryX import summary

def label2index(i):
    m = {'N':0, 'S':1, 'V':2, 'F':3, 'Q':4}
    return m[i]

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)

class CNN(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, out_channels, n_classes):
        super(CNN, self).__init__()
        
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # (batch, channels, length)
        self.cnn1 = nn.Conv1d(in_channels=self.in_channels, 
                            out_channels=self.out_channels, 
                            kernel_size=16, 
                            stride=2)
        self.cnn2 = nn.Conv1d(in_channels=self.out_channels, 
                            out_channels=self.out_channels, 
                            kernel_size=16, 
                            stride=2)
        self.cnn3 = nn.Conv1d(in_channels=self.out_channels, 
                            out_channels=self.out_channels, 
                            kernel_size=16, 
                            stride=2)
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.pool1 = nn.MaxPool1d(2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.pool2 = nn.MaxPool1d(2)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu3 = nn.ReLU()
        self.do3 = nn.Dropout(p=0.5)    
        self.pool3 = nn.MaxPool1d(2)    
        
        self.dense = nn.Linear(out_channels, n_classes)
        
    def forward(self, x):
        out = x
        
        out = self.cnn1(out)
        out = self.pool1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.do1(out)
        
        out = self.cnn2(out)
        out = self.pool2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.do2(out)
        
#         out = self.cnn3(out)
#         out = self.pool3(out)
#         out = self.bn3(out)
#         out = self.relu3(out)
#         out = self.do3(out)
        
        out = out.mean(-1)
        out = self.dense(out)

        return out
    
def resample(data, label):
    out_data = []
    out_label = []
    for i in range(len(label)):
        if label[i] == 0:
            if np.random.rand() > 0.9:
                out_data.append(data[i])
                out_label.append(label[i])
        else:
            out_data.append(data[i])
            out_label.append(label[i])
    return np.array(out_data), np.array(out_label)
    
    
if __name__ == "__main__":

    batch_size = 4096

    # make data
    print('start')
    data = np.load('data/mitdb_data.npy')
    data = preprocessing.scale(data, axis=1)
    print(data.shape)
    
    label_str = np.load('data/mitdb_label.npy')
    pid = np.load('data/mitdb_pid.npy')
    label = []
    for i in label_str:
        label.append(label2index(i))
    label = np.array(label)
    cnter = Counter(label)
    print(cnter)
    data, label = resample(data, label)
    cnter = Counter(label)
    print(cnter)
    weight = torch.Tensor([1, cnter[0]/cnter[1], cnter[0]/cnter[2], cnter[0]/cnter[3], cnter[0]/cnter[4]])
    print(weight)
    data = np.expand_dims(data, 1)

    # split train/val/test by pid, notice to avoid overlapping
    X_train, X_test, Y_train, Y_test = data, data, label, label
    print(X_train.shape, Y_train.shape)
    dataset = MyDataset(X_train, Y_train)
    dataset_test = MyDataset(X_test, Y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False, shuffle=True)
    
    # make model
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(device)
    model = CNN(
        in_channels=1, 
        out_channels=64, 
        n_classes=5)
    model.to(device)
    summary(model, torch.zeros(1, 1, 360))

    # train and test
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func = torch.nn.CrossEntropyLoss(weight=weight)

    n_epoch = 200
    step = 0
    prev_f1 = 0
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):

        # train
        model.train()
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):

            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
        
        print(loss.item())
        
        scheduler.step(_)
                    
        # test
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
        
        ## classification report
        print(classification_report(Y_test, all_pred))
        print(confusion_matrix(Y_test, all_pred))

