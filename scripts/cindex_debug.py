from src.dataset.dataset import HDFDataset, ValidationDataset
import torch
from src.transforms.transforms import (BaseTransform, RandomAmplifier, RandomGaussian, RandomPolarity,
                                       RandomTemporalScale, RandomShift, TanhNormalize)
from torchvision import transforms
from src.dataset.collate import collate_padding, collate_validation
from model.cox_mil_resnet import CoxAttentionResnet
from torchsurv.metrics.cindex import ConcordanceIndex
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

def plot_risks(preds, events):
    preds = preds.numpy()
    events = events.numpy()
    df = pd.DataFrame({'Risk': preds, 'Event': events})

    sns.histplot(data=df, x="Risk", hue="Event", bins = 10, multiple="stack", palette="vlag")
    plt.show()



if __name__ == "__main__":
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
    OVERSAMPLING_FACTOR = None
    GRAD_CLIP = 1.0

    MODEL_DIR = "/home/guest/lib/data/saved_models/cross_val_trial_27_fold3.pth"
    fold = 3
    batch_size = 24

    tanh_normalize = TanhNormalize(factor=5)
    val_transform = transforms.Compose([
            tanh_normalize,
        ])

    val_dataset = ValidationDataset(
            annotations_file=ANNOTATION_DIR,
            data_dir=DATA_DIR,
            eval_data=False,
            transform=val_transform,
            startswith="LA",
            readjustonce=True,
            segment_ms=SEGMENT_MS,
            filter_utilized=FILTER_UTILIZED,
            cross_val_fold=fold,
        )
    
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_validation)

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
        "hidden_size": 128,
        "attention_hidden_size": 64,
        "output_size": 1,
        "dropout": True,
        "dropout_prob": 0.5,
    }

    model = CoxAttentionResnet(resnet_params, amil_params)
    model.load_state_dict(torch.load(MODEL_DIR))
    model.eval()

    cindex = ConcordanceIndex()

    val_preds , val_durations, val_events = get_predictions(val_dataloader, model)
    concordance_val = cindex(estimate=val_preds.view(-1), event=val_events.view(-1), time=val_durations.view(-1))

    print(f"Concordance Index on Validation Set: {concordance_val:.4f}")

    plot_risks(val_preds, val_events)

    print("Čau, ahoj")