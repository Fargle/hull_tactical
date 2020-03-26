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
import json

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", help="load trained model", action="store_true")
    parser.add_argument("--train", help="train lstm model", action="store_true")
    parser.add_argument("--parameters", help="A json file containing a list of parameters to be used in the network", default={})
    parser.add_argument("--data", type=str, help="Path to data", default=os.path.abspath("ucsbdata.csv"))
    parser.add_argument("--disable-cuda", action="store_true", help="disables CUDA")
    return parser.parse_args()

class Model(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, n_layers, batch_size, seq_len, device, batch_first = True):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.LSTM = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.device = device
        self.cell_state = (torch.zeros(1,1,self.hidden_dim, device=device), 
                           torch.zeros(1,1,self.hidden_dim, device=device))
        
    def forward(self, in_seq):
        out, self.cell_state = self.LSTM(in_seq.view(len(in_seq), 1, -1), self.cell_state)
        predictions = self.linear(out.view(len(in_seq), -1)).to(device=self.device)       
        return predictions[-1]

def train(model, epochs, training_data, loss_function, optimizer, device):
    for i in range(epochs):
        for seq, label in adjusted_xnorm_train:
            optimizer.zero_grad()
            model.cell_state = (torch.zeros(1, 1, model.hidden_dim, device=device),
                                torch.zeros(1, 1, model.hidden_dim, device=device))
            seq = seq.to(device=device)
            y_pred = model(seq)

            label = label.to(device=device)
            single_loss = loss_function(y_pred, label)
            single_loss.backward()
            optimizer.step()

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
            print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

def validate(model, validation_data, loss_function, device):
    with open("results.csv", "a+") as results:
        result_writer = csv.writer(results, delimiter = ",", quotechar='"', quoting=csv.QUOTE_NONE)
        result_writer.writerow(["loss", "prediction", "actual"])

        for valid, label in validation_data:
            valid = valid.to(device=device)
            y_pred = model(valid)
            label = label.to(device=device)
            single_loss = loss_function(y_pred, label)

            result_writer.writerow([single_loss.item(), y_pred.item(), label.item()])

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
    sequence = int(sequence)
    adjusted_data = []
    L = data.size()[0]
    labels = torch.cat((labels, labels[len(labels)-1:]))
    for i in range(L - sequence):
        train_seq = data[i:i+sequence]
        train_label = labels[i+sequence+1]
        adjusted_data.append((train_seq, train_label.unsqueeze(0)))
    return adjusted_data

if __name__ == '__main__':
    args = setup()
    df = pd.read_csv(args.data) #save data into pandas data frame
    df = df[df["R"].notna()] #remove any rows where R does not have a value
    del df["Index"] #delete dates from our data 
    df = df.dropna() #need a different way to deal with nan  

    PATH = './model.pth'

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')

    try:
        parameters = json.load(open(args.parameters))
    except:
        parameters = args.parameters
    sequence_length =parameters.get("sequence length") if parameters.get("sequence length") is not None else   365
    batch_size =     parameters.get("batch size") if parameters.get("batch size") is not None else             365
    input_dim =      parameters.get("input dimension") if parameters.get("input dimension") is not None else   66
    out_dim =        parameters.get("output dimension") if parameters.get("output dimension") is not None else 1
    hidden_dim =     parameters.get("hidden dimension") if parameters.get("hidden dimension") is not None else 64
    layers =         parameters.get("layers") if parameters.get("layers") is not None else                     2
    lr =             parameters.get("learning rate") if parameters.get("learning rate") is not None else       0.001
    epochs =         parameters.get("epochs") if parameters.get("epochs") is not None else                     150

    model = Model(input_dim=input_dim, out_dim=out_dim, 
                  hidden_dim=hidden_dim, n_layers=layers, 
                  batch_size=batch_size, seq_len=sequence_length, 
                  device=args.device)
    model = model.to(device=args.device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if args.train:
        x_train, x_valid, train_labels, valid_labels = organize_data(df) #split our data into two sets, train and validate. 
        xnorm_train = normalize_data(x_train) #Normalize train and validation sets. 
        xnorm_valid = normalize_data(x_valid)
        trainnorm_labels = normalize_data(train_labels)
        validnorm_labels = normalize_data(valid_labels)
        print("Valid shape:", x_valid.shape, "Valid label:", valid_labels.shape)
        print("Train shape:", x_train.shape, "Train labels:", train_labels.shape)   

        xnorm_train = torch.FloatTensor(xnorm_train.values).view(-1, 66)
        trainnorm_labels = torch.FloatTensor(trainnorm_labels.values)
        xnorm_valid = torch.FloatTensor(xnorm_valid.values).view(-1, 66)
        validnorm_labels = torch.FloatTensor(validnorm_labels.values)

        print("xnorm torch tensor size:", xnorm_train.size(), "trainnorm size:", trainnorm_labels.size())
        adjusted_xnorm_train = create_sequences(xnorm_train, trainnorm_labels,  sequence_length)
        
        train(model, epochs, adjusted_xnorm_train, loss_function, optimizer, device=args.device)
        torch.save(model.state_dict(), PATH)
     
        adjusted_valid = create_sequences(xnorm_valid, validnorm_labels, sequence_length)
        validate(model, adjusted_valid, loss_function, device=args.device)
    
    if args.load:
        data_norm = normalize_data(df)
        labels = df["R"]
        del data_norm["R"]
        data_norm = torch.FloatTensor(data_norm.values).view(-1, 66)
        labels = torch.FloatTensor(labels.values)
        seq_norm = create_sequences(data_norm, labels, sequence_length)

        model.load_state_dict(torch.load(PATH))
        model.eval()
        validate(model, seq_norm, loss_function, device=args.device)
        

