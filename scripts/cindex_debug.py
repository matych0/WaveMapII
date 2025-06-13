from src.dataset.dataset import HDFDataset, ValidationDataset
import torch
from src.transforms.transforms import (BaseTransform, RandomAmplifier, RandomGaussian, RandomPolarity,
                                       RandomTemporalScale, RandomShift, TanhNormalize)
from torchvision import transforms
from src.dataset.collate import collate_padding_merged, collate_validation
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
    all_attentions = []
    all_traces = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            durations, events, traces, traces_masks = batch

            durations = torch.tensor(durations, dtype=torch.float32)
            events = torch.tensor(events, dtype=torch.bool)

            risks, attentions = model(traces, traces_masks)

            # Get rid of the padded traces and attentions
            #traces = traces[traces_masks]  
            #attentions = attentions[traces_masks]

            all_risks.extend([float(risk) for risk in risks])       
            all_durations.extend([int(duration) for duration in durations])
            all_events.extend([int(event) for event in events])
            all_attentions.extend([np.array(attention_array.squeeze().cpu()) for attention_array in attentions])
            all_traces.extend([trace.squeeze().cpu().numpy() for trace in traces])
            
    inference_df = pd.DataFrame({
        "risk": all_risks,
        "days_to_event": all_durations,
        "recurrence": all_events,
        "attentions": all_attentions,
        "traces": all_traces
    })

    risks_tensor = torch.tensor(all_risks, dtype=torch.float32).cpu()
    durations_tensor = torch.tensor(all_durations, dtype=torch.float32).cpu()
    events_tensor = torch.tensor(all_events, dtype=torch.bool).cpu()

    return risks_tensor, durations_tensor, events_tensor, inference_df

def plot_risks(inference_df):

    sns.histplot(data=inference_df, x="risk", hue="recurrence", bins=10, multiple="stack", palette="vlag")
    plt.show()


def plot_attention(inference_df, bins=30):
    if isinstance(inference_df, pd.DataFrame):
        attentions = np.concatenate(inference_df["attentions"].values, axis=0)
    elif isinstance(inference_df, pd.Series):
        attentions = inference_df["attentions"]
    g = sns.histplot(data=attentions, bins=bins)
    g.set_yscale("log")
    plt.title("Attention Weights Distribution")
    plt.xlabel("Attention Weight")
    plt.ylabel("Frequency")
    plt.show()


def plot_low_high_risk(inference_df, n_patients=10):
    low_risk_df = inference_df.nsmallest(n_patients, "risk")
    high_risk_df = inference_df.nlargest(n_patients, "risk")
    low_risk_attentions = np.concatenate(low_risk_df["attentions"].values, axis=0)
    high_risk_attentions = np.concatenate(high_risk_df["attentions"].values, axis=0)

    attentions_df = pd.DataFrame({
    'attention': np.concatenate([low_risk_attentions, high_risk_attentions]),
    'class': ['low risk'] * len(low_risk_attentions) + ['high risk'] * len(high_risk_attentions)
    })

    g = sns.histplot(data=attentions_df, x="attention", hue="class", bins=30, multiple="dodge", palette="vlag")
    g.set_yscale("log")
    plt.xlabel("Attention Weight")
    plt.ylabel("Frequency")
    plt.show()


def get_significant_traces(inference_df, threshold=0.01):
    """
    Returns traces with attention weights above a certain threshold.
    """
    all_attentions = np.concatenate(inference_df["attentions"].values, axis=0)
    all_traces = np.concatenate(inference_df["traces"].values, axis=0)
    significant_indices = np.where(all_attentions > threshold)[0]
    significant_traces = all_traces[significant_indices]
    significant_attentions = all_attentions[significant_indices]

    return significant_traces, significant_attentions

def plot_random_signals(signals, attention_weights):
    n = signals.shape[0]
    random_indices = np.random.choice(n, size=10, replace=False)

    # Select those 10 signals
    selected_signals = signals[random_indices]
    selected_attention_weights = attention_weights[random_indices]

    # Plot in a 5x2 grid
    fig, axes = plt.subplots(5, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(10):
        axes[i].plot(selected_signals[i])
        axes[i].set_title(f"Signal {random_indices[i]}, Attention: {selected_attention_weights[i]:.4f}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()



def plot_attentions(attentions):
    pass

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
    MODEL_DIR = f"/home/guest/lib/data/saved_models/cross_val_merged_2_fold_{fold}.pth"
    
    batch_size = 20

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
            eval_data=True,
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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_padding_merged)
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
    train_preds , train_durations, train_events, train_inference_df = get_predictions(train_cindex_dataloader, model)
    concordance_train = cindex(estimate=train_preds.view(-1), event=train_events.view(-1), time=train_durations.view(-1))

    val_preds , val_durations, val_events, val_inference_df = get_predictions(val_dataloader, model)
    concordance_val = cindex(estimate=val_preds.view(-1), event=val_events.view(-1), time=val_durations.view(-1))

    print(f"Concordance Index on Training Set: {concordance_train:.4f}")
    
    print(f"Concordance Index on Validation Set: {concordance_val:.4f}")

    plot_low_high_risk(train_inference_df, n_patients=10)
    plot_low_high_risk(val_inference_df, n_patients=10)

    significant_traces, significant_attentions = get_significant_traces(train_inference_df, threshold=0.01)
    print(f"Number of significant traces in training set: {significant_traces.shape[0]}")

    """ plot_risks(train_inference_df)
    plot_risks(val_inference_df)
    plot_attention(train_inference_df)
    plot_attention(val_inference_df) """

    print("Čau, ahoj")