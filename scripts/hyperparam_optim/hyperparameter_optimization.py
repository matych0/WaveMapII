import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataset.dataset import HDFDataset
from src.dataset.collate import collate_padding
from model.cox_mil_resnet import CoxAttentionResnet
from losses.loss import CoxLoss
from src.transforms.transforms import (RandomPolarity, RandomAmplifier, RandomGaussian,
                                       RandomTemporalScale, RandomShift, ZScore)
import datetime
import numpy as np


def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""

    # Hyperparameters to optimize
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    momentum = trial.suggest_uniform("momentum", 0.5, 0.99)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 3])
    num_epochs = 5

    # Dataset & Dataloader
    dataset = HDFDataset(
        annotations_file="/home/guest/lib/data/WaveMapSampleHDF/event_data.csv",
        data_dir="/home/guest/lib/data/WaveMapSampleHDF",
        train=True,
        transform=None,
        startswith="LA",
        readjustonce=True,
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

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_padding)
    
    # Model
    model = CoxAttentionResnet(resnet_params, amil_params)
    model.train()
    loss_fn = CoxLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    # TensorBoard logger
    log_dir = f"runs/optuna_experiment_{datetime.datetime.now().strftime('%d%m%Y-%H%M%S')}/optuna_trial_{trial.number}"
    writer = SummaryWriter(log_dir)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for case, control, case_mask, contrl_mask in dataloader:
            g_case, a_case = model(case, case_mask)
            g_control, a_control = model(control, contrl_mask)
            loss = loss_fn(g_case, g_control, shrink=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        scheduler.step()
    
    writer.add_hparams({"learning_rate": lr, "batch_size": batch_size, "momentum": momentum, "gamma": gamma}, {'hparam/loss': avg_loss})

    writer.close()
    return avg_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    print("Best hyperparameters:", study.best_params)