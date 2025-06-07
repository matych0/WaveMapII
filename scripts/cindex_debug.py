from src.dataset.dataset import HDFDataset, ValidationDataset
import torch
from src.transforms.transforms import (BaseTransform, RandomAmplifier, RandomGaussian, RandomPolarity,
                                       RandomTemporalScale, RandomShift, TanhNormalize)
from torchvision import transforms
from src.dataset.collate import collate_padding, collate_validation
from model.cox_mil_resnet import CoxAttentionResnet
from torchsurv.metrics.cindex import ConcordanceIndex
from torch.utils.data import DataLoader
from losses.loss import CoxLoss
import torch.optim as optim
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
    ANNOTATION_DIR = "/media/guest/DataStorage/WaveMap/HDF5/annotations_complete.csv"
    DATA_DIR = "/media/guest/DataStorage/WaveMap/HDF5"
    SEED = 3052001
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
    
    fold = 1
    MODEL_DIR = f"/home/guest/lib/data/saved_models/cross_val_complete_data{fold}.pth"
    
    batch_size = 10

    tanh_normalize = TanhNormalize(factor=5)
    val_transform = transforms.Compose([
            tanh_normalize,
        ])
    
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
    
    # Dataset & Dataloader
    train_dataset = HDFDataset(
        annotations_file=ANNOTATION_DIR,
        data_dir=DATA_DIR,
        train=False,
        transform=val_transform,
        startswith="LA",
        readjustonce=True,
        segment_ms=SEGMENT_MS,
        filter_utilized=FILTER_UTILIZED,
        #oversampling_factor=OVERSAMPLING_FACTOR,
        cross_val_fold=fold,
    )

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
    
    train_cindex_dataset = ValidationDataset(
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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_padding)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_validation)
    train_cindex_dataloader = DataLoader(train_cindex_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_validation)

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
        "dropout_prob": 0.2,
    }

    model = CoxAttentionResnet(resnet_params, amil_params)
    model.load_state_dict(torch.load(MODEL_DIR, map_location='cpu'))
    
    """ loss_fn = CoxLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.001)
    total_train_loss = 0.0
    model.eval()
    for case, control, case_mask, contrl_mask in train_dataloader:
        g_case, a_case = model(case, case_mask)
        g_control, a_control = model(control, contrl_mask)
        loss = loss_fn(g_case, g_control, shrink=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        print(f"loss: {loss:.4f}")

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss:.4f}") """

    cindex = ConcordanceIndex()

    model.eval()
    train_preds , train_durations, train_events = get_predictions(train_cindex_dataloader, model)
    concordance_train = cindex(estimate=train_preds.view(-1), event=train_events.view(-1), time=train_durations.view(-1))

    val_preds , val_durations, val_events = get_predictions(val_dataloader, model)
    concordance_val = cindex(estimate=val_preds.view(-1), event=val_events.view(-1), time=val_durations.view(-1))

    print(f"Concordance Index on Validation Set: {concordance_val:.4f}")

    plot_risks(train_preds, train_events)

    print("Čau, ahoj")