from src.dataset.dataset import HDFDataset
from src.dataset.collate import collate_padding
from model.cox_mil_resnet import CoxAttentionResnet
from losses.loss import CoxLoss
from src.transforms.transforms import RandomPolarity, RandomAmplifier, RandomGaussian, RandomTemporalScale, RandomShift, ZScore

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import datetime


# annotation_filepath = "C:/Users/matych/Research/SampleDataset/event_data.csv"
# dataset_folderpath = "C:/Users/matych/Research/SampleDataset"

annotation_filepath = "/home/guest/lib/data/WaveMapSampleHDF/event_data.csv"
dataset_folderpath = "/home/guest/lib/data/WaveMapSampleHDF"

seed = 42


polarity =  RandomPolarity(probability=0.5, shuffle=True, random_seed=seed)
temporal_scale = RandomTemporalScale(probability=0.5, limit=2, shuffle=True, random_seed=seed)
amplifier = RandomAmplifier(probability=0.5, limit=2, shuffle=True, random_seed=seed)
noise = RandomGaussian(probability=0.5, low_limit=10, high_limit=40, shuffle=True, random_seed=seed)
shift = RandomShift(probability=0.5, shuffle=True, random_seed=seed)
zcrore = ZScore(mean=1, std = 2)

training_data = HDFDataset(
    annotations_file=annotation_filepath,
    data_dir=dataset_folderpath,
    train=True,
    transform=transforms.Compose([
        polarity,
        temporal_scale,
        amplifier,
        noise,
        shift,
    ]),
    startswith="LA",
    readjustonce=True, 
    #num_traces=4000,
    segment_ms=100
)


# Define parameters for LocalActivationResNet
resnet_params = {
    "in_features": 1,
    "kernel_size": (1, 5),
    "stem_kernel_size": (1, 17),
    "blocks": [3,4,6,3],
    "features": [16, 32, 64, 128],
    "activation": "LReLU",
    "downsampling_factor": 4,
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
model = CoxAttentionResnet(resnet_params, amil_params)

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def main(save_model=True, save_plots=True):
    
    train_dataloader = DataLoader(training_data, batch_size=3, shuffle=True, collate_fn=collate_padding)
    loss_fn = CoxLoss()
    
    losses = []
    lrs = []

    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9955)
    num_epochs = 5   # Number of training epochs

    #torch.autograd.set_detect_anomaly(True)

    
    for epoch in range(num_epochs):
        total_loss = 0.0  # Track total loss for the epoch

        for case, control, case_mask, contrl_mask in train_dataloader:
            
            #x = torch.cat([case, control], dim=0)
            # Forward pass
            #g_y, a_y = model(x)
            
            g_case, a_case = model(case, case_mask)
            g_control, a_control = model(control, contrl_mask)
            #g_case = g_y[:3, :, :]
            #g_control = g_y[3:, :, :]
            
            # Compute loss
            loss = loss_fn(g_case, g_control, shrink=0)

            # Backpropagation
            optimizer.zero_grad()  # Reset gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights

            # Accumulate loss
            total_loss += loss.item()

        # Update learning rate
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        # Print loss for this epoch
        avg_loss = total_loss / len(train_dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    if save_model:    
        # Save the model
        save_dir = "C:/Users/matych/Research/WaveMap_trained/models/"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_filename = f"mil_cox_model_{timestamp}.pth"
        torch.save(model.state_dict(), save_dir + model_filename)
    
    if save_plots:
        # Plot the loss curve    
        smoothed_losses = moving_average(losses, window_size=10)
        
        plot_save_dir = "C:/Users/matych/Research/WaveMap_trained/plots/"
        plot_filename = f"training_loss{timestamp}.png"
        
        fig, ax = plt.subplots()
        ax.plot(range(1, num_epochs + 1), losses, label="Training Loss", color="blue")
        ax.plot(range(1, len(smoothed_losses) + 1), smoothed_losses, label="Smoothed Loss", color="red", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curve")
        ax.legend()
        fig.savefig(plot_save_dir + plot_filename)
        plt.show()
        
        # Plot the learning rate curve
        plot_filename = f"learning_rate{timestamp}.png"
        fig, ax = plt.subplots()
        ax.plot(range(1, num_epochs + 1), lrs, label="Learning Rate", color="green")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Curve")
        ax.legend()
        fig.savefig(plot_save_dir + plot_filename)
        plt.show()

if __name__ == "__main__":
    main(save_model=False, save_plots=False)