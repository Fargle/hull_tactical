import torch
import torch.nn as nn
import numpy as np
import pandas as pd 

class Model(nn.Module):
    def __init__(self):
        super(self, Model).__init__()
        self.conv1 = nn.Conv2d() 

if __name__ == '__main__':
    try:
        df = pd.read_csv("ucsbdata.csv") 
    except: 
        print('"results.csv": no such file')
    