# src/engine.py

import os
import torch
import torch.nn as nn
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


def freeze_bn(m):
    if isinstance(m, torch.nn.BatchNorm1d):
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # 1. Kaiming Normal is preferred for ReLU layers in the Encoder
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    # 2. Xavier/Glorot is better for the Attention layers (Tanh/Sigmoid)
    # We apply this specifically to the attention components
    if hasattr(m, 'attention_V') or hasattr(m, 'attention_U'):
        for layer in [m.attention_V, m.attention_U]:
            if isinstance(layer[0], nn.Linear):
                nn.init.xavier_uniform_(layer[0].weight)


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
        filter_center=cfg.data.dataset.filter_center,
        segment_ms=cfg.data.dataset.segment_ms,
        filter_utilized=cfg.data.dataset.filter_utilized,
        num_traces=cfg.data.dataset.num_traces,
        fold=fold,
        controls_time_gaussian_std=cfg.training.label_transforms.noise,
        random_seed=cfg.seed,
        shuffle_annotations=cfg.data.dataset.shuffle_annotations,
    )

    val_dataset = DatasetClass(
        annotations_file=cfg.data.paths.annotations_file,
        data_dir=cfg.data.paths.data_dir,
        startswith=cfg.data.dataset.startswith,
        training=False,
        transform=ValTransforms,
        readjustonce=cfg.data.dataset.readjustonce,
        filter_center=cfg.data.dataset.filter_center,
        segment_ms=cfg.data.dataset.segment_ms,
        filter_utilized=cfg.data.dataset.filter_utilized,
        num_traces=cfg.data.dataset.num_traces,
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
    batch_size = cfg.training.hparams.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ---- model ----
    ModelClass = hydra.utils.get_class(cfg.model.model_class)
    model = ModelClass().to(device) # cfg.model.resnet,cfg.model.mil_head
    model.apply(initialize_weights)

    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=model.parameters())
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.training.hparams.epochs * cfg.training.scheduler.warmup_fraction),
        num_training_steps=cfg.training.hparams.epochs,
    )

    # ---- task -----
    TaskClass = hydra.utils.get_class(cfg.training.task.handler)
    task = TaskClass(cfg, device)

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(cfg.training.hparams.epochs):

        # ===================== TRAIN =====================
        model.train()

        # Freeze batch norm layers
        # model.apply(freeze_bn)

        task.reset()

        total_train_loss = 0.0      # ← your original
        n_train_batches = 0         # correct averaging when some batches skipped

        for traces, masks, durations, events in train_loader:
            traces = traces.to(device)
            masks = masks.to(device)
            durations = durations.to(device)
            events = events.to(device)

            outputs, _ = model(traces, masks, batch_size=masks.size(0))

            loss = task.compute_loss(outputs, durations, events)

            # some tasks (survival) may skip batch
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()      # ← your original
            n_train_batches += 1

            task.update_train_metrics(outputs, durations, events)

        # ---- end train epoch ----

        if n_train_batches > 0:
            avg_train_loss = total_train_loss / n_train_batches   # ← your original (corrected)
        else:
            avg_train_loss = float("nan")

        scheduler.step()   # ← your original

        train_metrics = task.compute_epoch_train_metrics()

        # ===================== VALIDATION =====================
        model.eval()

        val_losses = []
        n_val_batches = 0

        with torch.no_grad():
            for traces, masks, durations, events in val_loader:
                traces = traces.to(device)
                masks = masks.to(device)
                durations = durations.to(device)
                events = events.to(device)

                outputs, att = model(traces, masks, batch_size=masks.size(0))

                #print(f"Max attention weight in batch: {att.max().item():.4f}")

                loss = task.compute_loss(outputs, durations, events)

                if loss is not None:
                    val_losses.append(loss.item())
                    n_val_batches += 1

                task.update_val_metrics(outputs, durations, events)

        if n_val_batches > 0:
            val_loss = sum(val_losses) / n_val_batches
        else:
            val_loss = float("nan")

        val_metrics = task.compute_epoch_val_metrics()

        # ===================== STORE HISTORY =====================
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        # store task-dependent metrics generically
        for k, v in train_metrics.items():
            history.setdefault(f"train_{k}", []).append(v)

        for k, v in val_metrics.items():
            history.setdefault(f"val_{k}", []).append(v)

        # ===================== OPTIONAL PRINT =====================
        metric_str = " | ".join(
            [f"train_{k}: {v:.4f}" for k, v in train_metrics.items()] +
            [f"val_{k}: {v:.4f}" for k, v in val_metrics.items()]
        )

        print(
            f"Epoch {epoch+1:03d} | "
            f"train_loss: {avg_train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"{metric_str}"
        )
    return {
        "history": history,
        "model": model,   # <-- return full model object
        "fold": fold,
    }

# -------------------------------------------------------------
# CV wrapper
# -------------------------------------------------------------

def run_training(cfg: DictConfig):
    set_seed(cfg.seed)

    results = []
    for fold in range(cfg.folds):
        results.append(train_one_fold(cfg, fold))
        #print(f"Fold {fold} finished: Validation C-index: {results[-1]['final_val_cindex']:.4f}")

    return results


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../config"):
        cfg: DictConfig = compose(config_name="config")

    cfg.training.hparams.batch_size = 16

    cfg.training.hparams.epochs = 100

    print(cfg)

    res = run_training(cfg)

    """ print("Final validation C-index for each fold:")
    for i, fold_result in enumerate(res):
        print(f"Fold {fold_result['fold']}: {fold_result['final_val_cindex']:.4f}") """