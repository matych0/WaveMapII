from src.dataset.dataset import ValidationDatasetInference
import torch
from src.transforms.transforms import TanhNormalize
from torchvision import transforms
from src.dataset.collate import collate_validation_inference
from model.cox_mil_resnet import CoxAttentionResnet
from torchsurv.metrics.cindex import ConcordanceIndex
from torch.utils.data import DataLoader
from losses.loss import CoxLoss
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from antropy import sample_entropy


def count_extrema_outside_band(signals, threshold=0.05):
    """
    Counts the number of local extrema (maxima and minima) in each signal that are
    outside of the ±threshold range around the baseline (mean of the signal).
    
    Parameters:
        signals (np.ndarray): Array of shape (N, fs), where each row is a signal.
        threshold (float): Threshold around the baseline in mV. Default is 0.05 mV.
        
    Returns:
        np.ndarray: Array of length N, each entry is the count of extrema outside the band.
    """
    extrema_counts = np.zeros(signals.shape[0], dtype=int)
    
    for i, signal in enumerate(signals):
        baseline = 0

        # Get indices of local maxima and minima
        maxima = argrelextrema(signal, np.greater)[0]
        minima = argrelextrema(signal, np.less)[0]

        # Combine extrema
        extrema_indices = np.concatenate((maxima, minima))
        extrema_values = signal[extrema_indices]

        # Check if extrema are outside the baseline ± threshold
        outside_mask = np.abs(extrema_values - baseline) > threshold
        extrema_counts[i] = np.sum(outside_mask)
    
    return extrema_counts


def compute_dominant_frequencies(signals, fs_hz):
    """
    Computes the dominant frequency for each 1D signal.

    Parameters:
        signals (np.ndarray): Shape (N, fs) with N signals.
        fs_hz (float): Sampling frequency in Hz.

    Returns:
        np.ndarray: Dominant frequency (in Hz) for each signal, shape (N,)
    """
    N, L = signals.shape
    dom_freqs = np.zeros(N)

    freqs = np.fft.rfftfreq(L, d=1/fs_hz)  # positive frequencies
    for i in range(N):
        fft_vals = np.fft.rfft(signals[i])
        magnitudes = np.abs(fft_vals)

        # Optionally skip DC component (freq = 0)
        dominant_idx = np.argmax(magnitudes[1:]) + 1
        dom_freqs[i] = freqs[dominant_idx]

    return dom_freqs


def get_predictions(loader, model):


    complete_inference_df = pd.DataFrame(columns=[
        "patient_path",
        "trace_id",
        "attention",
        "peak_to_peak",
        "extrema_count",
        "entropy"
        "dominant_frequency",
        #"feature_vector"
    ])

    model.eval()
    with torch.no_grad():
        for batch in loader:
            durations, events, traces, traces_masks, filepaths, trace_indices, peak_to_peaks = batch

            durations = torch.tensor(durations, dtype=torch.float32)
            events = torch.tensor(events, dtype=torch.bool)

            risks, attentions = model(traces, traces_masks)

            for i in range(len(durations)):
                patient_path = filepaths[i]
                trace_id = trace_indices[i]
                traces_mask = traces_masks[i].squeeze().cpu().numpy()
                egms = traces[i].squeeze().cpu().numpy()
                egms = egms[traces_mask == 1]  # Filter traces based on mask
                attention_array = attentions[i].squeeze().cpu().numpy()
                attention_array = attention_array[traces_mask == 1]
                #feature_vector = traces[i].squeeze().cpu().numpy()
                
                # 1) Calculate peak-to-peak for the filtered EGM
                ptp = peak_to_peaks[i]
                ptp_np = np.ptp(egms, axis=1)  # Peak-to-peak value of the EGM signal
                

                # 2) Count local extrema outside the ±0.1 mV band
                extrema_count = count_extrema_outside_band(egms, threshold=0.1)

                # 3) Calculate sample entropy for each EGM signal
                entropies = np.array([sample_entropy(sig) for sig in egms])

                # 4) Compute dominant frequencies
                dom_freqs = compute_dominant_frequencies(egms, fs_hz=2035)  # Assuming 2035 Hz sampling rate


                inference_df = pd.DataFrame({
                    "patient_path": [patient_path]* len(attention_array),
                    "trace_id": trace_id,
                    "attention": attention_array,
                    "peak_to_peak": ptp_np,
                    "extrema_count": extrema_count,
                    "entropy": entropies,
                    "dominant_frequency": dom_freqs,
                    #"feature_vector": feature_vector
                })

                # Append the predictions to the complete inference DataFrame
                complete_inference_df = pd.concat([complete_inference_df, inference_df], ignore_index=True)

    return complete_inference_df

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
    cindex = ConcordanceIndex()

    complete_inference_df = pd.DataFrame(columns=[
        "patient_path",
        "trace_id",
        "attention",
        "peak_to_peak",
        "extrema_count",
        "entropy",
        "dominant_frequency",
        #"feature_vector"
    ])


    for fold in range(3):
        
        val_dataset = ValidationDatasetInference(
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
        
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_validation_inference)

        model_dir_fold = f"/home/guest/lib/data/saved_models/cross_val_merged_2_fold_{fold}.pth"

        model = CoxAttentionResnet(resnet_params, amil_params)
        model.load_state_dict(torch.load(model_dir_fold, map_location='cpu'))

        model.eval()

        val_inference_df = get_predictions(val_dataloader, model)

        complete_inference_df = pd.concat([complete_inference_df, val_inference_df], ignore_index=True)
        #concordance_val = cindex(estimate=val_preds.view(-1), event=val_events.view(-1), time=val_durations.view(-1))

        #print(f"Concordance Index on Validation Set: {concordance_val:.4f}")

    # Save the complete inference DataFrame to a CSV file
    complete_inference_df.to_csv("/home/guest/lib/data/post_analysis/inference_results.csv", index=False)

    print("Čau, ahoj")