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

#hyperparameters
KERNEL_SIZE = (1, 5)
STEM_KERNEL_SIZE = (1, 17)
BLOCKS = [7, 7, 7, 7]
FEATURES = [16, 32, 64, 128]
ACTIVATION = "LReLU"
DOWNSAMPLING_FACTOR = 2
NORMALIZATION = "BatchN2D"


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
    projection_nodes = trial.suggest_categorical('projection_nodes', [64, 128, 256])
    attention_nodes = trial.suggest_categorical('attention_nodes', [32, 64, 128])
    dropout = trial.suggest_categorical('dropout', [0.0, 0.25, 0.5])
    cox_regularization = trial.suggest_categorical('cox_regularization', [0.0, 1e-3, 1e-2, 1e-1])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-4, 1e-3, 1e-2])


    batch_size = trial.suggest_categorical("batch_size", [1,2,3])
    num_epochs =  20 #trial.suggest_categorical("num_epochs", [50,100,200])

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
        "kernel_size": KERNEL_SIZE,
        "stem_kernel_size": STEM_KERNEL_SIZE,
        "blocks": BLOCKS,
        "features": FEATURES,
        "activation": ACTIVATION,
        "downsampling_factor": DOWNSAMPLING_FACTOR,
        "normalization": NORMALIZATION,
        "preactivation": False,
        "trace_stages": True,
    }

    # Define parameters for AMIL
    amil_params = {
        "input_size": FEATURES[-1],
        "hidden_size": projection_nodes,
        "attention_hidden_size": attention_nodes,
        "output_size": 1,
        "dropout": True,
        "dropout_prob": dropout,
    }

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_padding)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_validation)
    date = datetime.datetime.now().strftime('%d%m%Y-%H%M%S')


    # Create a directory for TensorBoard logs
    log_dir = f"runs/optuna_tryout_{date}/optuna_trial_{trial.number}"
    writer = SummaryWriter(log_dir)

    # Model, Loss, Optimizer
    model = CoxAttentionResnet(resnet_params, amil_params)
    loss_fn = CoxLoss()
    cindex = ConcordanceIndex()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0)
    
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
        writer.add_scalar("Train loss", avg_train_loss, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Learning Rate", current_lr, epoch)


        # Evaluation mode
        model.eval()
        val_preds , val_durations, val_events = get_predictions(val_dataloader, model)
        concordance_val = cindex(estimate=val_preds.view(-1), event=val_events.view(-1), time=val_durations.view(-1))
        writer.add_scalar("C-index evaluation", concordance_val, epoch)

        scheduler.step()

    # Log best values (e.g., final validation cindex and training loss)
    writer.add_hparams(
        {"projection_nodes": projection_nodes,
         "attention_nodes": attention_nodes,
         "dropout": dropout,
         "cox_regularization": cox_regularization,
         "learning_rate": learning_rate,
         "weight_decay": weight_decay,
         "batch_size": batch_size},
        {"hparam/concordance_val": concordance_val, 
         "hparam/loss": avg_train_loss}
    )

    writer.close()
    return concordance_val  # Maximize concordance → minimize negative


if __name__ == "__main__":

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)# n_startup_trials=10, )
    
    study = optuna.create_study(
        study_name="tuesday_tryout",
        storage="sqlite:///tuesday_tryout.db",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True
    )

    study.optimize(objective, n_trials=3)
    print("Best hyperparameters:", study.best_params)
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_slice(study)
    optuna.visualization.plot_parallel_coordinate(study)
    optuna.visualization.plot_contour(study)
    optuna.visualization.plot_edf(study)