from src.dataset.dataset import HDFDataset
from model.pyramid_resnet import LocalActivationResNet
from model.building_blocks import AttentionPooling
from losses.loss import CoxLoss

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader


class FullModel(nn.Module):
    def __init__(self, resnet_params, amil_params):
        super(FullModel, self).__init__()
        
        # Initialize the LocalActivationResNet
        self.resnet = LocalActivationResNet(**resnet_params)
        
        # Initialize the AMIL layer
        self.amil = AttentionPooling(**amil_params)

    def forward(self, x):
        # Pass input through ResNet
        x = self.resnet(x)
        
        # Transpose dimensions for compatibility
        x = x.transpose(-2, -1)
        
        # Pass through AMIL
        risk, a = self.amil(x)
        
        return risk, a

# Define parameters for LocalActivationResNet
resnet_params = {
    "in_features": 1,
    "kernel_size": (1, 5),
    "stem_kernel_size": (1, 17),
    "blocks": [9, 12, 18, 9],
    "features": [16, 32, 64, 128],
    "activation": "LReLU",
    "normalization": "BatchN2D",
    "preactivation": False,
    "trace_stages": True
}

# Define parameters for AMIL
amil_params = {
    "input_size": 128,
    "hidden_size": 128,
    "attention_hidden_size": 64,
    "output_size": 1,
    "dropout": False,
    "dropout_prob": 0.25
}

# Instantiate the full model
model = FullModel(resnet_params, amil_params)

annotation_filepath = "C:/Users/matych/Desktop/SampleDataset/event_data.csv"
dataset_folderpath = 'C:/Users/matych/Desktop/SampleDataset'

training_data = HDFDataset(
    annotations_file=annotation_filepath,
    data_dir=dataset_folderpath,
    train=True,
    transform=None,            
    startswith="LA",
    readjustonce=False, 
    num_traces=4000          
)

train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True)


""" for i in range(20):"""
case, control = next(iter(train_dataloader)) 



g_case, a_case = model(case)
g_control, a_control = model(control)

loss_fn = CoxLoss()
loss = loss_fn(g_case, g_control, shrink=0.1)

print(loss)
    
    