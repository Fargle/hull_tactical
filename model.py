import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import pandas as pd 
from sklearn import preprocessing

class Model(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, n_layers, batch_size, seq_len, batch_first = True):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.LSTM = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim) 
        self.cell_state = (torch.zeros(1,1,self.hidden_dim), 
                           torch.zeros(1,1,self.hidden_dim))
        
    def forward(self, in_seq):
        out, self.cell_state = self.LSTM(in_seq.view(len(in_seq), 1, -1), self.cell_state)
        predictions = self.linear(out.view(len(in_seq), -1))
        return predictions[-1]

#Splits our data into two sets. Default is 80%, 20% split. 
def organize_data(df, train = .8):
    mask = int(len(df)*train)
    x_train = df[~mask:]
    train_labels = x_train["R"]
    x_valid = df[mask:]
    valid_labels = x_valid["R"]

    x_train = x_train.drop("R", axis=1)
    x_valid = x_valid.drop("R", axis=1)
    return x_train, x_valid, train_labels, valid_labels

#Normalize by taking the mean and standard deviation of our dataset. 
def normalize_data(data):
    def normalize(x, mean, std):
        return (x - mean)/std
    mean, std = data.mean(), data.std()
    return normalize(data, mean, std)

def create_sequences(data, labels, sequence):
    adjusted_data = []
    L = data.size()[0]
    labels = torch.cat((labels, labels[len(labels)-1:]))
    for i in range(L - sequence):
        train_seq = data[i:i+sequence]
        train_label = labels[i+sequence+1]
        adjusted_data.append((train_seq, train_label.unsqueeze(0)))
    print(train_label.shape)
    return adjusted_data
    

if __name__ == '__main__':
    PATH = os.path.abspath("ucsbdata.csv")#get path to data
    df = pd.read_csv(PATH) #save data into pandas data frame
    df = df[df["R"].notna()] #remove any rows where R does not have a value
    del df["Index"] #delete dates from our data 
    ##df = df.fillna(0)
    x_train, x_valid, train_labels, valid_labels = organize_data(df) #split our data into two sets, train and validate. 
    xnorm_train = normalize_data(x_train) #Normalize train and validation sets. 
    xnorm_valid = normalize_data(x_valid)
    trainnorm_labels = normalize_data(train_labels)
    validnorm_labels = normalize_data(valid_labels)
    print("Valid shape:", x_valid.shape, "Valid label:", valid_labels.shape)
    print("Train shape:", x_train.shape, "Train labels:", train_labels.shape)

    #print("Train:", x_train.iloc[:5], "xnorm_train: ", xnorm_train.iloc[:5])
    #print("Valid:", x_valid.iloc[:5], "xnorm_valid: ", xnorm_valid.iloc[:5])
    xnorm_train = torch.FloatTensor(xnorm_train.values).view(-1, 66)
    trainnorm_labels = torch.FloatTensor(trainnorm_labels.values)
    xnorm_valid = torch.FloatTensor(xnorm_valid.values).view(-1, 66)
    validnorm_labels = torch.FloatTensor(validnorm_labels.values)

    print("xnorm torch tensor size:", xnorm_train.size(), "trainnorm size:", trainnorm_labels.size())
    #print("length of xnorm train:", len(xnorm_train))

    sequence_length = 365
    adjusted_xnorm_train = create_sequences(xnorm_train, trainnorm_labels, sequence_length)
    batch_size = 100
    input_dim = 66
    output_dim = 1
    hidden_dim = 64
    n_layers = 2


    #print(adjusted_xnorm_train[:5])
    model = Model(input_dim = input_dim, out_dim=output_dim, hidden_dim=hidden_dim, n_layers=n_layers, batch_size=1, seq_len = sequence_length)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 150

  
    for i in range(epochs):
        for seq, label in adjusted_xnorm_train:
            optimizer.zero_grad()
            model.cell_state = (torch.zeros(1, 1, model.hidden_dim),
                                torch.zeros(1, 1, model.hidden_dim))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, label)
            single_loss.backward()
            optimizer.step()

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
            print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')