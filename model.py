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
import wandb


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", help="syncs data with weights and biases remote", action="store_true")
    parser.add_argument("--load", help="load trained model", action="store_true")
    parser.add_argument("--train", help="train lstm model", action="store_true")
    parser.add_argument("--parameters", help="A json file containing a list of parameters to be used in the network", default={})
    parser.add_argument("--data", type=str, help="Path to data", default=os.path.abspath("ucsbdata.csv"))
    parser.add_argument("--disable-cuda", action="store_true", help="disables CUDA")
    parser.add_argument("--name", help="name the model.pth file", default='model')
    return parser.parse_args()


class Model(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, n_layers, batch_size, seq_len, dropout, device):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.indicator_dim = 8
        self.input_dim = input_dim
        self.layers = n_layers
        self.batch_size = batch_size
        self.LSTM = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.deep = nn.Linear((hidden_dim + self.indicator_dim)*seq_len, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.device = device
        self.cell_state = (torch.zeros(self.layers, self.batch_size ,self.hidden_dim, device=device), 
                           torch.zeros(self.layers, self.batch_size ,self.hidden_dim, device=device))
        
    def forward(self, in_seq, indicators):
        out, self.cell_state = self.LSTM(in_seq.view(self.batch_size, len(in_seq[0]), -1), self.cell_state)
        indicators = indicators.view(self.batch_size, 1, -1)
        indicators = indicators.expand(self.batch_size, len(out[0]), self.indicator_dim)
        out = torch.cat((out, indicators), dim=2).to(device=self.device)
        deep = self.deep(out.view(-1, (self.hidden_dim+self.indicator_dim)*len(in_seq[0]))).to(device=self.device)
        deep = self.dropout(deep)
        predictions = self.linear(deep.view(self.batch_size, self.hidden_dim)).to(device=self.device)
        return predictions[-1]


def train(model, epochs, training_data, loss_function, optimizer, device, sync = False):
    for i in range(epochs):
        for batch, labels, indicators in training_data:
            #model.zero_grad()
            optimizer.zero_grad()
            model.cell_state = (torch.zeros(model.layers, model.batch_size, model.hidden_dim, device=device),
                                torch.zeros(model.layers, model.batch_size, model.hidden_dim, device=device))
            batch = batch.to(device=device)
            indicators  = indicators.to(device=device)
            y_pred = model(batch, indicators)
            labels = labels.to(device=device)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i%25 == 1:
            loss = single_loss.sum()
            ave_loss = single_loss.sum()/batch_size
            if sync:
                wandb.log({"Test Accuracy": ave_loss, "Test Loss": loss})
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
            print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


def validate(model, validation_data, loss_function, device, filename, sync):
    with open(filename, "a+") as results:
        result_writer = csv.writer(results, delimiter = ",", quotechar='"', quoting=csv.QUOTE_NONE, escapechar='\\')
        result_writer.writerow(["loss", "prediction", "actual"])

        for valid, label in validation_data:
            valid = valid.to(device=device)
            y_pred = model(valid)
            label = label.to(device=device)
            single_loss = loss_function(y_pred, label)
            
            if sync:
                wandb.log({"Test Loss": single_loss.item()})

            for i in range(len(label)-1):
                result_writer.writerow([single_loss.item(), y_pred.data[i][0].tolist(), label.data[i][0].tolist()])


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


#Calculates the simple moving average given a time period and list of sequences.
def moving_ave(sequence, time_period):
    #sequences in a list of tuples containing the time window and the corresponding label.
    moving_ave = torch.sum(sequence, axis=0)/time_period
    return moving_ave
    
    
#this function takes the list of simple moving averages and returns the exponential moving average
#s is the smoothing. 
#sequences is the list of tuples containing the time window and the corresponding label.
#value_t is todays stock value.
def exp_moving_ave(sequence, time_period, prev_ema = None, s = 2):
    price_today = sequence[len(sequence)-1][4:8]
    #print('len sequence', len(sequence))
    #print('price today', price_today)
    if prev_ema is None:
        prev_ema = moving_ave(sequence[0:time_period, 3:7], time_period=len(sequence))
        #print('prev ema', prev_ema)
    weight_multiplier = s/float(1+time_period)
    ema = ((price_today - prev_ema) * weight_multiplier) + prev_ema
    return ema


def append_indicators(sequences, time_period):
    appended = []
    ema = exp_moving_ave(sequences[0][0], time_period=time_period)
    for seq, label in sequences:
        indicators = moving_ave(seq[:, 3:7], len(seq))
        ema = exp_moving_ave(seq, time_period=time_period)
        indicators = torch.cat((indicators, ema), dim=-1)
        appended.append((seq, label, indicators))
    return appended


def create_batches(sequences, batch_size):
    assert(batch_size <= len(sequences))
    batched_data = []
    for x in range(len(sequences)-batch_size):
        if  x%batch_size == 0:
            batch = [[i for i, j, k in sequences[x:x+batch_size]],
                     [j for i, j, k in sequences[x:x+batch_size]],
                     [k for i, j, k in sequences[x:x+batch_size]]]
            batched_data.append((torch.stack(batch[0]), torch.stack(batch[1]), torch.stack(batch[2])))
    return batched_data


if __name__ == '__main__':
    args = setup()
    df = pd.read_csv(args.data) #save data into pandas data frame
    df = df[df["R"].notna()] #remove any rows where R does not have a value
    del df["Index"] #delete dates from our data 
    df = df.dropna() #need a different way to deal with nan  

    model_name = args.name + '.pth'

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')

        
    try:
        parameters = json.load(open(args.parameters))
    except:
        print("unable to load params.json, using default")
        parameters = args.parameters
    sequence_length =parameters.get("sequence length") if parameters.get("sequence length") is not None else   60
    batch_size =     parameters.get("batch size") if parameters.get("batch size") is not None else             1
    input_dim =      parameters.get("input dimension") if parameters.get("input dimension") is not None else   66
    out_dim =        parameters.get("output dimension") if parameters.get("output dimension") is not None else 1
    hidden_dim =     parameters.get("hidden dimension") if parameters.get("hidden dimension") is not None else 64
    layers =         parameters.get("layers") if parameters.get("layers") is not None else                     2
    lr =             parameters.get("learning rate") if parameters.get("learning rate") is not None else       0.001
    epochs =         parameters.get("epochs") if parameters.get("epochs") is not None else                     200
    dropout =        parameters.get("dropout") if parameters.get("dropout") is not None else                   0.2
    time_period =    parameters.get("time period") if parameters.get("time period") is not None else           20

    model = Model(input_dim=input_dim, out_dim=out_dim, 
                  hidden_dim=hidden_dim, n_layers=layers, 
                  batch_size=batch_size, seq_len=sequence_length, 
                  dropout = dropout, device=args.device)
    model = model.to(device=args.device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if args.sync:
        wandb.init(project='hull-tactical')
        wandb.watch(model)

    outfile = args.name + "-results.csv"
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
        appended_xnorm_train = append_indicators(adjusted_xnorm_train, time_period=time_period)
        batched_xnorm_train = create_batches(appended_xnorm_train, batch_size)
        
        train(model, epochs, batched_xnorm_train, loss_function, optimizer, device=args.device, sync=args.sync)
        if args.sync:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, model_name)) 
        torch.save(model.state_dict(), model_name)
     
        adjusted_valid = create_sequences(xnorm_valid, validnorm_labels, sequence_length)
        appended_valid = append_indicators(adjusted_valid, time_period=time_period)
        batched_valid = create_batches(adjusted_valid, batch_size)
        validate(model, batched_valid, loss_function, device=args.device, filename=outfile, sync=args.sync)
    
    if args.load:
        data_norm = normalize_data(df)
        labels = data_norm["R"]
        del data_norm["R"]
        data_norm = torch.FloatTensor(data_norm.values).view(-1, 66)
        labels = torch.FloatTensor(labels.values)
        seq_norm = create_sequences(data_norm, labels, sequence_length)
        batches = create_batches(seq_norm, batch_size)

        model.load_state_dict(torch.load(model_name, map_location=args.device))
        model.eval()
        validate(model, batches, loss_function, device=args.device, filename=outfile, sync=args.sync)
        
