import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataset.dataset import HDFDataset, ValidationDataset
from src.dataset.collate import collate_padding, collate_validation
from model.cox_mil_resnet import CoxAttentionResnet
from losses.loss import CoxLoss
from src.transforms.transforms import (RandomPolarity, RandomAmplifier, RandomGaussian,
                                       RandomTemporalScale, RandomShift, ZScore)
import datetime
import numpy as np

from pycox.evaluation import EvalSurv
from torchsurv.metrics.cindex import ConcordanceIndex
import torchtuples as tt
import pandas as pd


ANNOTATION_DIR = "/home/guest/lib/data/WaveMapSampleHDF/event_data.csv"
DATA_DIR = "/home/guest/lib/data/WaveMapSampleHDF"


def get_predictions(loader, model):
    all_risks, all_durations, all_events = [], [], []
    with torch.no_grad():
        for batch in loader:
            durations, events, traces, traces_masks = batch
            risks, attentions = model(traces, traces_masks)
            all_risks.append(risks.squeeze().cpu().numpy())
            all_durations.append(durations)  # Adjust based on how durations/events are stored
            all_events.append(events)        # Adjust accordingly
    return torch.tensor(np.array(all_risks)), torch.tensor(np.array(all_durations), dtype=torch.float), torch.tensor(np.array(all_events), dtype=torch.bool)


def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""

    # Hyperparameters to optimize
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    momentum = trial.suggest_uniform("momentum", 0.5, 0.99)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 3])
    num_epochs = 10

    # Dataset & Dataloader
    train_dataset = HDFDataset(
        annotations_file=ANNOTATION_DIR,
        data_dir=DATA_DIR,
        train=True,
        transform=None,
        startswith="LA",
        readjustonce=True,
        segment_ms=100
    )

    val_dataset = ValidationDataset(
        annotations_file=ANNOTATION_DIR,
        data_dir=DATA_DIR,
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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_padding)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_validation)
    date = datetime.datetime.now().strftime('%d%m%Y-%H%M%S')


    # Create a directory for TensorBoard logs
    log_dir = f"runs/optuna_tryout_{date}/optuna_trial_{trial.number}"
    writer = SummaryWriter(log_dir)

    # Model, Loss, Optimizer
    model = CoxAttentionResnet(resnet_params, amil_params)
    loss_fn = CoxLoss()
    cindex = ConcordanceIndex()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    model.train()

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        model.train()
        for case, control, case_mask, contrl_mask in train_dataloader:
            g_case, a_case = model(case, case_mask)
            g_control, a_control = model(control, contrl_mask)
            loss = loss_fn(g_case, g_control, shrink=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # Evaluation mode
        model.eval()
        val_preds , val_durations, val_events = get_predictions(val_dataloader)
        concordance_val = cindex(estimate=val_preds.view(-1), event=val_events.view(-1), time=val_durations.view(-1))
        writer.add_scalar("Eval/concordance_val", concordance_val, epoch)

        scheduler.step()

    # Log best values (e.g., final validation cindex and training loss)
    writer.add_hparams(
        {"learning_rate": lr, "batch_size": batch_size, "momentum": momentum, "gamma": gamma},
        {'hparam/concordance_val': concordance_val, 'hparam/loss': avg_train_loss}
    )

    writer.close()
    return concordance_val  # Maximize concordance → minimize negative


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)
    print("Best hyperparameters:", study.best_params)