#!/usr/bin/env python

import os
import json
import ember
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor
import numpy as np

class EmberDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return torch.Tensor(feature), torch.tensor(label, dtype=torch.long)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def train_model(datadir, featureversion):
    # Load the training data
    X_train_path = os.path.join(datadir, "X_train.dat")
    y_train_path = os.path.join(datadir, "y_train.dat")
    X_train = ember.read_vectorized_features(X_train_path, featureversion)
    # y_train = ember.read_labels(y_train_path)
    y_train = np.loadtxt(y_train_path, dtype=np.int32, delimiter=',', skiprows=1, usecols=0,encoding='latin-1')

    # Create a PyTorch DataLoader
    train_dataset = EmberDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Create the CNN model and optimizer
    model = CNNModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    model.train()
    for epoch in range(10):  # You can adjust the number of epochs
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{10}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(datadir, "model.pt"))


def main():
    prog = "train_ember"
    descr = "Train an Ember model from a directory with raw feature files"

    datadir = "../data/ember2018"  # Set your desired default path here

    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("-v", "--featureversion", type=int, default=2, help="EMBER feature version")
    # parser.add_argument("datadir", metavar="DATADIR", type=str, default=default_datadir, help="Directory with raw features")
    args = parser.parse_args()

    # if not os.path.exists(datadir) or not os.path.isdir(datadir):
    #     parser.error("{} is not a directory with raw feature files".format(datadir))

    X_train_path = os.path.join(datadir, "X_train.dat")
    y_train_path = os.path.join(datadir, "y_train.dat")
    print(os.path.exists("../data/ember2018/X_train.dat"))
    print(os.path.exists(y_train_path))
    

    # if not (os.path.exists(X_train_path) and os.path.exists(y_train_path)):
    #     print("Creating vectorized features")
    #     ember.create_vectorized_features(datadir, args.featureversion)

    print("Training CNN model")
    train_model(datadir, args.featureversion)


if __name__ == "__main__":
    main()
