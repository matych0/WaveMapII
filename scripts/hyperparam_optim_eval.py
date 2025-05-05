import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataset.dataset import HDFDataset, ValidationDataset
from src.dataset.collate import collate_padding, collate_validation
from model.cox_mil_resnet import CoxAttentionResnet
from losses.loss import CoxLoss
from src.transforms.transforms import (BaseTransform, RandomAmplifier, RandomGaussian,
                                       RandomTemporalScale, RandomShift, TanhNormalize)
import datetime
import numpy as np

from pycox.evaluation import EvalSurv
from torchsurv.metrics.cindex import ConcordanceIndex
import torchtuples as tt
import pandas as pd
from statistics import median


#Set seed
SEED = 3052001

ANNOTATION_DIR = "/media/guest/DataStorage/WaveMap/HDF5/annotations_train.csv"
DATA_DIR = "/media/guest/DataStorage/WaveMap/HDF5"

#hyperparameters
KERNEL_SIZE = (1, 5)
STEM_KERNEL_SIZE = (1, 17)
BLOCKS = [3, 4, 6, 3]
FEATURES = [16, 32, 64, 128]
ACTIVATION = "LReLU"
DOWNSAMPLING_FACTOR = 2
NORMALIZATION = "BatchN2D"
FILTER_UTILIZED = True
SEGMENT_MS = 100
OVERSAMPLING_FACTOR = 4


def get_predictions(loader, model):
    all_risks = []
    all_durations = []
    all_events = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            durations, events, traces, traces_masks = batch

            durations = torch.tensor(durations, dtype=torch.float32)
            events = torch.tensor(events, dtype=torch.bool)

            risks, _ = model(traces, traces_masks)

            all_risks.append(risks.view(-1))        # Instead of .squeeze()
            all_durations.append(durations.view(-1))
            all_events.append(events.view(-1))

    risks_tensor = torch.cat(all_risks).cpu()
    durations_tensor = torch.cat(all_durations).cpu()
    events_tensor = torch.cat(all_events).cpu()

    return risks_tensor, durations_tensor, events_tensor


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""
    set_seed(SEED)


    # Hyperparameters to optimize
    projection_nodes = trial.suggest_categorical('projection_nodes', [64, 128, 256])
    attention_nodes = trial.suggest_categorical('attention_nodes', [32, 64, 128])
    dropout = trial.suggest_categorical('dropout', [0.0, 0.25, 0.5])
    cox_regularization = trial.suggest_categorical('cox_regularization', [0.0, 1e-3, 1e-2, 1e-1])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 1e-3, 5e-3, 1e-2, 5e-2])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-4, 1e-3, 1e-2])
    batch_size =  trial.suggest_categorical("batch_size", [4,8,16,24])
    num_epochs =  trial.suggest_int("num_epochs", 50, 500)



    # Transformations
    temporal_scale = RandomTemporalScale(probability=0.2, limit=0.2, shuffle=True, random_seed=SEED)
    amplifier = RandomAmplifier(probability=0.2, limit=0.2, shuffle=True, random_seed=SEED)
    noise = RandomGaussian(probability=0.2, low_limit=10, high_limit=30, shuffle=True, random_seed=SEED)
    shift = RandomShift(probability=0.5, shift_range=0.3, shuffle=True, random_seed=SEED)
    tanh_normalize = TanhNormalize(factor=5)
    shuffle = BaseTransform(shuffle=True, random_seed=SEED)

    train_transform = transforms.Compose([
        amplifier,
        temporal_scale,
        shift,
        noise,
        shuffle,
        tanh_normalize,
    ])

    val_transform = transforms.Compose([
        tanh_normalize,
    ])

    # Dataset & Dataloader
    train_dataset = HDFDataset(
        annotations_file=ANNOTATION_DIR,
        data_dir=DATA_DIR,
        train=True,
        transform=train_transform,
        startswith="LA",
        readjustonce=True,
        segment_ms=SEGMENT_MS,
        filter_utilized=FILTER_UTILIZED,
        oversampling_factor=OVERSAMPLING_FACTOR,
    )

    val_dataset = ValidationDataset(
        annotations_file=ANNOTATION_DIR,
        data_dir=DATA_DIR,
        transform=val_transform,
        startswith="LA",
        readjustonce=True,
        segment_ms=SEGMENT_MS,
        filter_utilized=FILTER_UTILIZED,
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


    # Create DataLoader
    generator = torch.Generator()
    generator.manual_seed(SEED)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=collate_padding)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_validation)
    date = datetime.datetime.now().strftime('%d%m%Y-%H%M%S')


    # Create a directory for TensorBoard logs
    log_dir = f"runs/hyperparam_optim_02_05/optuna_trial_{trial.number}"
    writer = SummaryWriter(log_dir)

    # Model, Loss, Optimizer
    model = CoxAttentionResnet(resnet_params, amil_params)
    loss_fn = CoxLoss()
    cindex = ConcordanceIndex()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_epochs // 10,
        num_training_steps=num_epochs, # total number of steps (warmup + cosine decay)
    )

    val_concordances = []
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        model.train()
        for case, control, case_mask, contrl_mask in train_dataloader:
            g_case, a_case = model(case, case_mask)
            g_control, a_control = model(control, contrl_mask)
            loss = loss_fn(g_case, g_control, shrink=cox_regularization)

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
        val_concordances.append(concordance_val)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation C-index: {concordance_val:.4f}")
        writer.add_scalar("C-index evaluation", concordance_val, epoch)

        scheduler.step()

    median_concordance = median(val_concordances[-10:])

    # Log best values (e.g., final validation cindex and training loss)
    writer.add_hparams(
        {"projection_nodes": projection_nodes,
         "attention_nodes": attention_nodes,
         "dropout": dropout,
         "cox_regularization": cox_regularization,
         "weight_decay": weight_decay,
         "batch_size": batch_size,
         "learning_rate": learning_rate,
         "num_epochs": num_epochs},
        {"hparam/concordance_val": median_concordance, 
         "hparam/loss": avg_train_loss}
    )

    writer.close()

    return median_concordance  # Maximize concordance → minimize negative


if __name__ == "__main__":

    num_trials = 100

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=3052001, n_startup_trials=20)
    
    study = optuna.create_study(
        study_name="hyperparam_optim_02_05",
        storage="sqlite:///hyperparam_optim_02_05.db",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True
    )

    study.optimize(objective, n_trials=num_trials)
    print("Best hyperparameters:", study.best_params)

    """ optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_slice(study)
    optuna.visualization.plot_parallel_coordinate(study)
    optuna.visualization.plot_contour(study)
    optuna.visualization.plot_edf(study) """