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

from pycox.evaluation import EvalSurv
import torchtuples as tt
import pandas as pd

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
    date = datetime.datetime.now().strftime('%d%m%Y-%H%M%S')

    log_dir = f"runs/optuna_tryout_cindex/optuna_trial_{trial.number}"

    # Split into training and validation (for example purposes, use 80/20 split)
    """ n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_padding)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_padding)
 """
    model = CoxAttentionResnet(resnet_params, amil_params)
    model.train()
    loss_fn = CoxLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        model.train()
        for case, control, case_mask, contrl_mask in dataloader:
            g_case, a_case = model(case, case_mask)
            g_control, a_control = model(control, contrl_mask)
            loss = loss_fn(g_case, g_control, shrink=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # Evaluation mode
        model.eval()

        def get_predictions(loader):
            preds, durations, events = [], [], []
            with torch.no_grad():
                for batch in loader:
                    case, _, case_mask, _ = batch
                    g_case, _ = model(case, case_mask)
                    preds.append(g_case.squeeze().cpu().numpy())
                    durations.append(case[0]['duration'])  # Adjust based on how durations/events are stored
                    events.append(case[0]['event'])        # Adjust accordingly
            return np.array(preds), np.array(durations), np.array(events)

        val_preds, val_durations, val_events = get_predictions(val_loader)
        train_preds, train_durations, train_events = get_predictions(train_loader)

        # Create survival predictions
        val_surv_df = pd.DataFrame(val_preds, columns=["score"])
        train_surv_df = pd.DataFrame(train_preds, columns=["score"])

        # EvalSurv needs sorted durations
        val_idx = np.argsort(-val_durations)
        train_idx = np.argsort(-train_durations)

        ev_val = EvalSurv(pd.DataFrame(-val_surv_df.values.T), val_durations[val_idx], val_events[val_idx], censor_surv='km')
        ev_train = EvalSurv(pd.DataFrame(-train_surv_df.values.T), train_durations[train_idx], train_events[train_idx], censor_surv='km')

        concordance_val = ev_val.concordance_td()
        concordance_train = ev_train.concordance_td()

        writer.add_scalar("Eval/concordance_val", concordance_val, epoch)
        writer.add_scalar("Eval/concordance_train", concordance_train, epoch)

        scheduler.step()

    # Log best values (e.g., final validation loss)
    writer.add_hparams(
        {"learning_rate": lr, "batch_size": batch_size, "momentum": momentum, "gamma": gamma},
        {'hparam/concordance_val': concordance_val, 'hparam/loss': avg_train_loss}
    )

    writer.close()
    return concordance_val  # Maximize concordance → minimize negative


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    print("Best hyperparameters:", study.best_params)