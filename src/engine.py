# src/engine.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchsurv.metrics.cindex import ConcordanceIndex
from transformers import get_cosine_schedule_with_warmup

import hydra
from hydra.utils import get_object
from omegaconf import DictConfig


# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_cases_controls(risks, events, durations, n_controls):
    """Sampling identical to avg_pooling.py and best_trial.py."""
    device = risks.device

    risks = risks.view(-1)
    events = events.view(-1)
    durations = durations.view(-1)

    g_cases = risks[events]
    if g_cases.numel() == 1:
        g_cases = g_cases.reshape(1)

    n_cases = g_cases.numel()
    g_controls = torch.zeros(n_controls, n_cases, device=device)

    for i in range(n_cases):
        case_time = durations[events][i]
        valid_controls = risks[durations >= case_time]
        idx = torch.randint(0, len(valid_controls), (n_controls,), device=device)
        g_controls[:, i] = valid_controls[idx]

    return g_cases, g_controls


# -------------------------------------------------------------
# Dataset + loaders
# -------------------------------------------------------------

def build_datasets(cfg, fold):
    DatasetClass = hydra.utils.get_class(cfg.data.dataset.dataset.class_name)
    TrainTransforms = hydra.utils.instantiate(cfg.data.transforms.train)
    ValTransforms = hydra.utils.instantiate(cfg.data.transforms.val)

    train_dataset = DatasetClass(
        annotations_file=cfg.data.paths.annotations_file,
        data_dir=cfg.data.paths.data_dir,
        startswith=cfg.data.dataset.startswith,
        training=True,
        transform=TrainTransforms,
        readjustonce=cfg.data.dataset.readjustonce,
        segment_ms=cfg.data.dataset.segment_ms,
        filter_utilized=cfg.data.dataset.filter_utilized,
        fold=fold,
        controls_time_shift=cfg.training.label_transforms.shift,
        controls_time_gaussian_std=cfg.training.label_transforms.noise,
        random_seed=cfg.seed,
    )

    val_dataset = DatasetClass(
        annotations_file=cfg.data.paths.annotations_file,
        data_dir=cfg.data.paths.data_dir,
        startswith=cfg.data.dataset.startswith,
        training=False,
        transform=ValTransforms,
        readjustonce=cfg.data.dataset.readjustonce,
        segment_ms=cfg.data.dataset.segment_ms,
        filter_utilized=cfg.data.dataset.filter_utilized,
        fold=fold,
        random_seed=cfg.seed,
    )

    return train_dataset, val_dataset


# -------------------------------------------------------------
# One training fold
# -------------------------------------------------------------

def train_one_fold(cfg, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- datasets ----
    train_dataset, val_dataset = build_datasets(cfg, fold)
    collate_fn = hydra.utils.get_object(cfg.data.dataset.dataset.collate_fn)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.hparams.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.hparams.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ---- model ----
    ModelClass = hydra.utils.get_class(cfg.model.model_class)
    model = ModelClass(cfg.model.resnet, cfg.model.mil_head).to(device)

    # ---- optimizer + loss ----
    loss_fn = hydra.utils.instantiate(cfg.training.loss).to(device)

    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=model.parameters())
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.training.hparams.epochs * cfg.training.scheduler.warmup_fraction),
        num_training_steps=cfg.training.hparams.epochs,
    )

    cindex = ConcordanceIndex()

    # ---- training ----
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_cindex": [],
        "val_cindex": [],
    }

    n_controls = cfg.training.hparams.n_controls

    for epoch in range(cfg.training.hparams.epochs):

        model.train()
        total_train_loss = 0.0

        epoch_risks = []
        epoch_events = []
        epoch_durations = []

        for traces, masks, durations, events in train_loader:
            traces = traces.to(device)
            masks = masks.to(device)
            durations = durations.to(device)
            events = events.to(device)

            if not events.any():
                continue

            else:
                risks, _ = model(traces, masks)
                g_cases, g_controls = sample_cases_controls(risks, events, durations, n_controls)

                optimizer.zero_grad()
                loss = loss_fn(g_cases, g_controls)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                epoch_risks.append(risks.detach().cpu().view(-1))
                epoch_events.append(events.detach().cpu().view(-1))
                epoch_durations.append(durations.detach().cpu().view(-1))

        # ---- epoch end ----
        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step()

        epoch_risks = torch.cat(epoch_risks)
        epoch_events = torch.cat(epoch_events)
        epoch_durations = torch.cat(epoch_durations)

        train_cidx = cindex(estimate=epoch_risks, event=epoch_events, time=epoch_durations)

        # ---- validation ----
        model.eval()
        val_losses = []
        val_preds = []
        val_events_list = []
        val_durations_list = []

        with torch.no_grad():
            for traces, masks, durations, events in val_loader:
                traces = traces.to(device)
                masks = masks.to(device)
                durations = durations.to(device)
                events = events.to(device)

                if not events.any():
                    continue

                risks, _ = model(traces, masks)
                g_cases, g_controls = sample_cases_controls(risks, events, durations, n_controls)

                val_losses.append(loss_fn(g_cases, g_controls).item())
                val_preds.append(risks.detach().cpu().view(-1))
                val_events_list.append(events.cpu().view(-1))
                val_durations_list.append(durations.cpu().view(-1))

        val_loss = np.mean(val_losses)
        val_preds = torch.cat(val_preds)
        val_events_cat = torch.cat(val_events_list)
        val_durations_cat = torch.cat(val_durations_list)

        val_cidx = cindex(estimate=val_preds, event=val_events_cat, time=val_durations_cat)

        # ---- store metrics ----
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["train_cindex"].append(train_cidx)
        history["val_cindex"].append(val_cidx)

    # Save model
    save_path = os.path.join(cfg.data.paths.save_model_dir, f"{cfg.study_name}_{cfg.experiment_name}_{cfg.model.name}_fold_{fold}.pth")
    torch.save(model.state_dict(), save_path)

    return {
        "history": history,
        "final_val_cindex": val_cidx,
        "model_path": save_path,
    }


# -------------------------------------------------------------
# CV wrapper
# -------------------------------------------------------------

def run_training(cfg: DictConfig):
    set_seed(cfg.seed)

    results = []
    for fold in range(cfg.folds):
        results.append(train_one_fold(cfg, fold))

    return results
