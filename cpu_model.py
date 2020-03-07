import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import pandas as pd 
import csv
import argparse

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", help="load trained model", action="store_true")
    parser.add_argument("--train", help="train lstm model", action="store_true")
    parser.add_argument("--data", type=str, help="Path to data", default=os.path.abspath("ucsbdata.csv"))    
    return parser.parse_args()

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


def validate(model, validation_data, loss_function):
    with open("results.csv", "a+") as results:
        result_writer = csv.writer(results, delimiter = ",", quotechar='"', quoting=csv.QUOTE_NONE)
        result_writer.writerow(["loss", "prediction", "actual"])

        for valid, label in validation_data:
            y_pred = model(valid)
            single_loss = loss_function(y_pred, label)

            result_writer.writerow([single_loss.item(), y_pred.item(), label.item()])
    return y_pred.item()

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
    return adjusted_data
    

if __name__ == "__main__": 
    args = setup()
    df = pd.read_csv(args.data) #save data into pandas data frame
    df = df[df["R"].notna()] #remove any rows where R does not have a value
    del df["Index"] #delete dates from our data 
    df = df.dropna() #need a different way to deal with nan  
    
    PATH = './model.pth'
    sequence_length = 365
    batch_size = 365
    input_dim = 66
    output_dim = 1
    hidden_dim = 128
    n_layers = 3
    model = Model(input_dim = input_dim, out_dim=output_dim, hidden_dim=hidden_dim, n_layers=n_layers, batch_size=1, seq_len = sequence_length)
    loss_function = nn.MSELoss()

    if args.load:
        data_norm = normalize_data(df)
        labels = df["R"]
        del data_norm["R"]
        data_norm = torch.FloatTensor(data_norm.values).view(-1, 66)
        labels = torch.FloatTensor(labels.values)
        seq_norm = create_sequences(data_norm, labels, sequence_length)

        model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        model.eval()
        prediction = validate(model, seq_norm, loss_function)
        print(prediction)