import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import pandas as pd 

class Model(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, n_layers, batch_size, seq_len, batch_first = True):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.LSTM = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first)
        self.linear = nn.Linear(hidden_dim, out_dim) 
        self.cell_state = (torch.zeros(1,1,self.hidden_dim), 
                           torch.zeros(1,1,self.hidden_dim))
        
    def forward(self, in_seq):
        out, self.cell_state = self.lstm(in_seq.view(len(in_seq), 1, -1), self.cell_state)
        predictions = self.linear(out.view(len(in_seq), -1))
        return predictions[-1]

#Splits our data into two sets. Default is 80%, 20% split. 
def organize_data(df, train = .8):
    mask = int(len(df)*train)
    x_train = df[~mask:]
    x_valid = df[mask:]
    return x_train, x_valid

#Normalize by taking the mean and standard deviation of our dataset. 
def normalize_data(data):
    def normalize(x, mean, std):
        return (x - mean)/std
    mean, std = data.mean(), data.std()
    return normalize(data, mean, std)

def create_sequences(data, sequence):
    adjusted_data = []
    L = len(data)
    for i in range(L - sequence):
        train_seq = data[i:i+sequence]
        train_label = data[i+sequence: i+sequence+1]
        adjusted_data.append((train_seq, train_label))
    return adjusted_data

if __name__ == '__main__':
    PATH = os.path.abspath("ucsbdata.csv")#get path to data
    df = pd.read_csv(PATH) #save data into pandas data frame
    df = df[df["R"].notna()] #remove any rows where R does not have a value
    del df["Index"] #delete dates from our data 
    #df = df.fillna(0)
    x_train, x_valid = organize_data(df) #split our data into two sets, train and validate. 
    xnorm_train = normalize_data(x_train) #Normalize train and validation sets. 
    xnorm_valid = normalize_data(x_valid)
    print("Valid shape:", x_valid.shape)
    print("Train shape:", x_train.shape)
    print("Train:", x_train.iloc[:5], "xnorm_train: ", xnorm_train.iloc[:5])
    print("Valid:", x_valid.iloc[:5], "xnorm_valid: ", xnorm_valid.iloc[:5])
    xnorm_train = torch.FloatTensor(xnorm_train.values).view(-1)
    xnorm_valid = torch.FloatTensor(xnorm_valid.values).view(-1)
    print("length of xnorm train:", len(xnorm_train))

    sequence_length = 365
    adjusted_xnorm_train = create_sequences(xnorm_train, sequence_length)
    print(adjusted_xnorm_train[:5])
    model = Model(66, 1, 100, 2, batch_size=64, seq_len = sequence_length)


