import os
import time
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms

# 超参数

TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
SEED = 4396

LENGTH = 512
WIDTH, HEIGHT = 32, 16
BATCH_SIZE = 16
EPOCH = 300
SHUFFLE = False
CLASSES = 2

LR = 1e-4


datapath = "/home/jovyan/histogram"

with open("/home/datacon/malware/XXX/black.txt", 'r') as f:
    black_list = f.read().strip().split()

with open("/home/datacon/malware/XXX/white.txt", 'r') as f:
    white_list = f.read().strip().split()

black_path = [os.path.join(datapath, sp) for sp in black_list]
white_path = [os.path.join(datapath, sp) for sp in white_list]

raw_feature, raw_labels = [], []

with tqdm(total=11647, ncols=80, desc="histogram") as pbar:
    for fp in black_path:
        with open(fp+'.txt', 'r') as f:
            feature = f.readlines()
        feature = [float(his.strip()) for his in feature]
        raw_feature.append(feature)
        raw_labels.append(1)
        pbar.update(1)
    for fp in white_path:
        with open(fp+'.txt', 'r') as f:
            feature = f.readlines()
        feature = [float(his.strip()) for his in feature]
        raw_feature.append(feature)
        raw_labels.append(0)
        pbar.update(1)

# 打乱顺序

np.random.seed(SEED)
torch.manual_seed(SEED)

features, labels = np.array(raw_feature, dtype=np.float32), np.array(raw_labels, dtype=np.int32)

index = list(range(len(labels)))
np.random.shuffle(index)

features = features[index]
labels = labels[index] 

# 划分数据集

train_features, test_features, train_label, test_label = train_test_split(
    features,
    labels,
    test_size=TEST_SIZE,
    stratify=labels,
    random_state=SEED)
train_features, valid_features, train_label, valid_label = train_test_split(
    train_features,
    train_label,
    test_size=VAL_SIZE,
    stratify=train_label,
    random_state=SEED)

# 加载dataset

class HistogramDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label

train_ds = HistogramDataset(train_features, train_label)
valid_ds = HistogramDataset(valid_features, valid_label)
test_ds = HistogramDataset(test_features, test_label)

train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
valid_loader = data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
test_loader = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=SHUFFLE)


# 模型

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 60, kernel_size=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(60, 200, kernel_size=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(200*8*4, 500)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(500, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 1, WIDTH, HEIGHT)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, 200*8*4)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

criterion = nn.BCELoss()
optimizer = optim.Nadam(model.parameters(), lr=LR)

for epoch in range(EPOCH):
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    
    valid_loss = 0.0
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            valid_loss += loss.item()
    valid_loss /= len(valid_loader)
    
    print('[Epoch %d] train loss: %.3f, validation loss: %.3f' % (epoch+1, train_loss, valid_loss))
    
    if epoch > 5 and valid_loss > max(valid_losses[-5:]):
        print('Early stopping')
        break
    
    valid_losses.append(valid_loss)

predict = 0.0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        predict += loss.item()
predict /= len(test_loader)
print('Test loss: %.3f' % predict)

torch.save(model.state_dict(), './models/histogram_{0:.2f}.pth'.format(predict))