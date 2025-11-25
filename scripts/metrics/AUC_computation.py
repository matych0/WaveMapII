import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from src.dataset.dataset import EGMDataset
from src.dataset.collate import collate_padding
from model.cox_mil_resnet import CoxAttentionResnet
from src.transforms.transforms import (TanhNormalize)
import numpy as np
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
import os

FOLDS = 3
MODEL_BASE_DIR = "D:/saved_models/AB_testing"
MODEL_NAME = "trial_127_reproduction"
MODEL_FOLD_DIRS = [os.path.join(MODEL_BASE_DIR, f"{MODEL_NAME}_fold_{i}.pth") for i in range(FOLDS)]

SEED = 3052001

ANNOTATION_DIR = "D:/HDF5/annotations_complete.csv"
DATA_DIR = "D:/HDF5"

SEGMENT_MS = 100
FILTER_UTILIZED = True

#hyperparameters
KERNEL_SIZE = (1, 5)
STEM_KERNEL_SIZE = (1, 17)
BLOCKS = [3, 4, 6, 3]
FEATURES = [16, 32, 64, 128]
PROJECTION_NODES = 128
ATTENTION_NODES = 64
ACTIVATION = "LReLU"
DOWNSAMPLING_FACTOR = 2
NORMALIZATION = "BatchN2D"
DROPOUT = 0.5

BATCH_SIZE = 32
CHUNK_SIZE = 8

AUC_TIMES = torch.tensor([182, 364], dtype=torch.float32)  # 6 months and 12 months in days

tanh_normalize = TanhNormalize(factor=5)

# CHANGE TRANSFOR FOR MIN-MAX NORMALIZATION
val_transform = transforms.Compose([
            tanh_normalize,
        ])

cindex = ConcordanceIndex()

auc = Auc()

for fold, model_fold_dir in enumerate(MODEL_FOLD_DIRS):
    # CHANGE FOR LABEL SHIFT AND LABEL NOISE
    val_dataset = EGMDataset(
                annotations_file=ANNOTATION_DIR,
                data_dir=DATA_DIR,
                startswith="LA",
                training=False,
                transform=val_transform,
                readjustonce=True,
                segment_ms=SEGMENT_MS,
                filter_utilized=FILTER_UTILIZED,
                #oversampling_factor=OVERSAMPLING_FACTOR,
                #controls_time_shift=60,  # Shift controls by 60 days
                fold=fold,
                random_seed=SEED,
            )

    val_batch_size = BATCH_SIZE if BATCH_SIZE <= CHUNK_SIZE else CHUNK_SIZE
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_padding)

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

    # Define parameters for MAXMIL
    mil_params = {
        "input_size": FEATURES[-1],
        "hidden_size": PROJECTION_NODES,
        "attention_hidden_size": ATTENTION_NODES,
        "output_size": 1,
        "dropout": True,
        "dropout_prob": DROPOUT,
    }

    model = CoxAttentionResnet(resnet_params, mil_params)

    model.load_state_dict(torch.load(model_fold_dir, map_location=torch.device("cpu")))
    
    print(f"Evaluating model: {MODEL_NAME}, fold {fold}...")
    
    model.eval()
    
    with torch.no_grad():
        val_risks, val_events, val_durations = list(), list(), list()
        for traces, masks, durations, events in val_dataloader:
            risks, _ = model(traces, masks)
            val_risks.append(risks.view(-1))
            val_events.append(events.view(-1))
            val_durations.append(durations.view(-1))
    
            
    val_risks = torch.cat(val_risks, dim=0)
    val_events = torch.cat(val_events, dim=0)
    val_durations = torch.cat(val_durations, dim=0)
    concordance_val = cindex(estimate=val_risks, event=val_events, time=val_durations)
    
    auc_val = auc(estimate=val_risks, event=val_events, time=val_durations, new_time=AUC_TIMES)

    print(f"Fold {fold}, Validation C-index: {concordance_val:.4f}")
    
    print(f"Fold {fold}, Validation AUC at 6 months: {auc_val[0]:.4f}")
    
    print(f"Fold {fold}, Validation AUC at 12 months: {auc_val[1]:.4f}")

    