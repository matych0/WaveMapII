import numpy as np
import pandas as pd
import torch
from antropy import sample_entropy
from scipy.signal import argrelextrema
from torch.utils.data import DataLoader
from torchvision import transforms

from model.cox_mil_resnet import CoxAttentionResnet
from src.dataset.collate import collate_validation_inference
from src.dataset.dataset import ValidationDatasetInference
from src.transforms.transforms import TanhNormalize


def count_extrema_outside_band(signals, threshold=0.1):
    """
    Counts the number of local extrema (maxima and minima) in each signal that are
    outside of the ±threshold range around the baseline.
    
    Parameters:
        signals (np.ndarray): Array of shape (N, fs), where each row is a signal.
        threshold (float): Threshold around the baseline in mV. Default is 0.1 mV.
        
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
        signals (np.ndarray): Shape (M, N) with M signals.
        fs_hz (float): Sampling frequency in Hz.

    Returns:
        np.ndarray: Dominant frequency (in Hz) for each signal, shape (M,)
    """
    N, L = signals.shape
    dom_freqs = np.zeros(N)

    freqs = np.fft.rfftfreq(L, d=1/fs_hz)  # positive frequencies
    for i in range(N):
        fft_vals = np.fft.rfft(signals[i])
        magnitudes = np.abs(fft_vals)

        # Skip DC component (freq = 0)
        dominant_idx = np.argmax(magnitudes[1:]) + 1
        dom_freqs[i] = freqs[dominant_idx]

    return dom_freqs


def get_predictions(loader, model):


    complete_inference_df = pd.DataFrame(columns=[
        "patient_path",
        "risk",
        "trace_id",
        "attention",
        "peak_to_peak",
        "extrema_count",
        "entropy",
        "dominant_frequency",
        "feature_vector",
        "traces"
    ])

    model.eval()
    with torch.no_grad():
        for batch in loader:
            durations, events, traces, traces_orig, traces_masks, filepaths, trace_indices, peak_to_peaks = batch

            durations = torch.tensor(durations, dtype=torch.float32)
            events = torch.tensor(events, dtype=torch.bool)

            risks, (features, attentions) = model(traces, traces_masks)

            for i in range(len(durations)):
                patient_path = filepaths[i]
                risk = risks[i].item()
                trace_id = trace_indices[i]
                traces_mask = traces_masks[i].squeeze().cpu().numpy()
                egms = traces_orig[i].squeeze().cpu().numpy()
                egms = egms[traces_mask == 1]  # Filter traces based on mask
                attention_array = attentions[i].squeeze().cpu().numpy()
                attention_array = attention_array[traces_mask == 1]
                feature_vectors = features[i].squeeze().cpu().numpy()
                feature_vectors = feature_vectors[traces_mask == 1]
                
                # 1) Calculate peak-to-peak for the filtered EGM
                ptp_np = np.ptp(egms, axis=1)  # Peak-to-peak value of the EGM signal
                
                # 2) Count local extrema outside the ±0.1 mV band
                extrema_count = count_extrema_outside_band(egms, threshold=0.1)

                # 3) Calculate sample entropy for each EGM signal
                entropies = np.array([sample_entropy(sig) for sig in egms])

                # 4) Compute dominant frequencies
                dom_freqs = compute_dominant_frequencies(egms, fs_hz=2035)  # Assuming 2035 Hz sampling rate

                inference_df = pd.DataFrame({
                    "patient_path": [patient_path]* len(attention_array),
                    "risk": [risk]* len(attention_array),
                    "trace_id": trace_id,
                    "attention": attention_array,
                    "peak_to_peak": ptp_np,
                    "extrema_count": extrema_count,
                    "entropy": entropies,
                    "dominant_frequency": dom_freqs,
                    #"feature_vector": [feature_vectors[i] for i in range(len(feature_vectors))],
                    #"traces": [egms[i] for i in range(len(egms))],
                })

                complete_inference_df = pd.concat([complete_inference_df, inference_df], ignore_index=True)

    return complete_inference_df


if __name__ == "__main__":
    DF_STORAGE_PATH = ""
    MODEL_DIR = ""

    ANNOTATION_DIR = ""
    DATA_DIR = ""
    SEED = 3052001
    
    # hyperparameters
    KERNEL_SIZE = (1, 5)
    STEM_KERNEL_SIZE = (1, 17)
    BLOCKS = [3, 4, 6, 3]
    FEATURES = [16, 32, 64, 128]
    PROJECTION_NODES = 128
    ATTENTION_NODES = 64
    ACTIVATION = "LReLU"
    DOWNSAMPLING_FACTOR = 2
    NORMALIZATION = "BatchN2D"
    FILTER_UTILIZED = True
    SEGMENT_MS = 100
    OVERSAMPLING_FACTOR = None
    CHUNK_SIZE = 8
    FOLDS = 3
    
    
    batch_size = 8

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
        "hidden_size": PROJECTION_NODES,
        "attention_hidden_size": ATTENTION_NODES,
        "output_size": 1,
        "dropout": True,
        "dropout_prob": 0.2,
    }

    complete_inference_df = pd.DataFrame(columns=[
        "patient_path",
        "risk",
        "trace_id",
        "attention",
        "peak_to_peak",
        "extrema_count",
        "entropy",
        "dominant_frequency",
        "feature_vector",
        "traces"
    ])


    for fold in range(FOLDS):
        
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

        model_dir_fold = f"{MODEL_DIR}_fold_{fold}.pth"

        model = CoxAttentionResnet(resnet_params, amil_params)
        model.load_state_dict(torch.load(model_dir_fold, map_location='cpu'))

        model.eval()

        val_inference_df = get_predictions(val_dataloader, model)
        
        print(f"Fold {fold} - Inference DataFrame shape: {val_inference_df.shape}")

        complete_inference_df = pd.concat([complete_inference_df, val_inference_df], ignore_index=True)

    # Save the complete inference DataFrame to a CSV file
    inference_df_export = complete_inference_df[["patient_path", "trace_id", "attention", "peak_to_peak", "extrema_count", "entropy", "dominant_frequency"]]
    inference_df_export.to_csv(DF_STORAGE_PATH, index=False)