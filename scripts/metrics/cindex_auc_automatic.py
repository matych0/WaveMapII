import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from src.dataset.dataset import EGMDataset
from src.dataset.collate import collate_padding
from model.cox_mil_resnet import CoxAttentionResnet, CoxMaxResnet, CoxAvgResnet
from src.transforms.transforms import (TanhNormalize, Normalize)
import numpy as np
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
import os
import pandas as pd
import glob


FOLDS = 3
MODEL_BASE_DIR = "D:/saved_models/AB_testing"
MODEL_NAME = "avg_pooling"  # Change this to the model name you want to evaluate

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
min_max_normalize = Normalize()

# CHANGE TRANSFOR FOR MIN-MAX NORMALIZATION
val_transform = transforms.Compose([
            tanh_normalize,
        ])
        
""" val_transform = transforms.Compose([
            min_max_normalize,
        ]) """

cindex = ConcordanceIndex()

auc = Auc()

#model_paths = glob.glob(os.path.join(MODEL_BASE_DIR, "*.pth"))
model_paths = glob.glob(os.path.join(MODEL_BASE_DIR, f"{MODEL_NAME}_fold_*.pth"))

model_paths = sorted(model_paths)

results = []

for model_path in model_paths:
    model_filename = os.path.basename(model_path)  # e.g., trial_127_reproduction_fold_0.pth
    fold = int(model_filename.split("_fold_")[-1].split(".")[0])  # extract fold number

    val_dataset = EGMDataset(
        annotations_file=ANNOTATION_DIR,
        data_dir=DATA_DIR,
        startswith="LA",
        training=True,
        transform=val_transform,
        readjustonce=True,
        segment_ms=SEGMENT_MS,
        filter_utilized=FILTER_UTILIZED,
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

    mil_params = {
        "input_size": FEATURES[-1],
        "hidden_size": PROJECTION_NODES,
        #"attention_hidden_size": ATTENTION_NODES,
        "output_size": 1,
        "dropout": True,
        "dropout_prob": DROPOUT,
    }

    #model = CoxAttentionResnet(resnet_params, mil_params)
    #model = CoxMaxResnet(resnet_params, mil_params)
    model = CoxAvgResnet(resnet_params, mil_params)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        val_risks, val_events, val_durations = list(), list(), list()
        for traces, masks, durations, events in val_dataloader:
            #risks, _ = model(traces, masks)
            risks = model(traces, masks)
            val_risks.append(risks.view(-1))
            val_events.append(events.view(-1))
            val_durations.append(durations.view(-1))

    val_risks = torch.cat(val_risks, dim=0)
    val_events = torch.cat(val_events, dim=0)
    val_durations = torch.cat(val_durations, dim=0)

    concordance_val = cindex(estimate=val_risks, event=val_events, time=val_durations)
    auc_val = auc(estimate=val_risks, event=val_events, time=val_durations, new_time=AUC_TIMES)

    # Save metrics in results list
    results.append({
        "model_file": model_filename,
        "c_index": float(concordance_val),
        "auc_6_months": float(auc_val[0]),
        "auc_12_months": float(auc_val[1]),
    })
    
    print(f"Model: {model_filename}, C-Index: {concordance_val:.4f}, AUC 6 months: {auc_val[0]:.4f}, AUC 12 months: {auc_val[1]:.4f}")

# Convert to DataFrame and export to CSV
results_df = pd.DataFrame(results)
results_csv_path = f"C:/Users/marti/Documents/Diplomka/results/cindex_auc_{MODEL_NAME}_training.csv"
results_df.to_csv(results_csv_path, index=False)

print(f"Saved results to {results_csv_path}")

