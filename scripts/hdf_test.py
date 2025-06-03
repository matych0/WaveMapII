import os
import h5py
import pandas as pd
import torch
import numpy as np
import seaborn as sns
from typing import Union
import matplotlib.pyplot as plt

def read_hdf(
        file_path: Union[str, os.PathLike],
        return_fs: bool = False,
        metadata_keys: list = None,
    ):  
    with h5py.File(file_path, 'r') as hf:
        traces = hf["traces"][:]    
        if metadata_keys:
            fs = hf["traces"].attrs["sampling_frequency"]
            metadata = {key: hf[key][:] for key in metadata_keys}
            return traces, fs, metadata
        elif return_fs:
            fs = hf["traces"].attrs["sampling_frequency"]
            return traces, fs
        else:
            return traces
        
def compute_amplitude_indices(t_amp, t_last, fs, num_samples):
    """
    Computes the sample indices where amplitude events occur.

    Parameters:
    t_amp (np.ndarray): Array of Unix timestamps when amplitude events occur.
    t_last (np.ndarray): Array of Unix timestamps of the last sample.
    fs (float): Sampling frequency in Hz.
    num_samples (int): Number of samples in the signal.

    Returns:
    np.ndarray: Array of sample indices for each amplitude event.
    """
    # Ensure inputs are numpy arrays and convert byte strings if necessary
    t_amp = np.array([float(x.decode()) if isinstance(x, bytes) else float(x) for x in t_amp])
    t_last = np.array([float(x.decode()) if isinstance(x, bytes) else float(x) for x in t_last])

    # Compute the time of the first sample for each signal
    t_first = t_last - (num_samples - 1) / fs

    # Compute sample indices (rounded to nearest integer)
    sample_indices = np.round((t_amp - t_first) * fs).astype(int)

    return sample_indices


def segment_signals(signals, indices, segment_ms, fs):
    """
    Extracts segments of predefined length around a central index for each signal.
    """
    segment_samples = np.round((segment_ms/1000)*fs).astype(int)
    left_window = segment_samples//2
    right_window = segment_samples - left_window

    # Initialize output array
    N, T = signals.shape
    segments = np.zeros((N, segment_samples))

    for i in range(N):
        central_sample = indices[i]
        start, end = (int(max(central_sample - left_window, 0)), int(min(central_sample + right_window, T)))

        # Handle cases where the segment would be cut off at the beginning or end
        segment = signals[i, start:end]

        # If the extracted segment is shorter than expected, pad with zeros
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')

        segments[i] = segment

    return segments

# Main processing function
def scan_hdf_directories(base_dir: str) -> pd.DataFrame:
    rows = []

    for root, dirs, files in os.walk(base_dir):
        hdf_files = [f for f in files if f.endswith(".hdf")]

        if hdf_files:
            subfolder_name = os.path.basename(root)
            hdf_count = len(hdf_files)

            try:
                first_hdf_path = os.path.join(root, hdf_files[0])
                traces, fs, metadata = read_hdf(first_hdf_path, return_fs=True, metadata_keys=["utilized"])
                shape = traces.shape
                utilized = metadata["utilized"]
            except Exception as e:
                print(f"Failed to read {first_hdf_path}: {e}")
                shape = None
                fs = None
                utilized = None

            rows.append({
                "EnSiteID": subfolder_name,
                "shape": shape,
                "fs": fs,
                "hdf_file_count": hdf_count,
                "orig_instances": shape[0] if shape is not None else None,
                "utilized": np.sum(utilized) if utilized is not None else None,
            })
    
    df = pd.DataFrame(rows)
    num_instances = [shape[0] for shape in df["shape"]]
    min_num_instances = min(num_instances)
    max_num_instances = max(num_instances)
    mean_num_instances = sum(num_instances) / len(num_instances)
    print(num_instances)
    print(f"Min: {min_num_instances}, Max: {max_num_instances}, Mean: {mean_num_instances}")
    
    # Save the DataFrame to a CSV file
    df.to_csv("/media/guest/DataStorage/WaveMap/WaveMapEnsiteAnnotations/hdf_dataset_overview_overall.csv", index=False)

    return df


def plot_instances_histogram(df: pd.DataFrame, bins=20):

    df = df.melt(value_vars=["orig_instances", "utilized"], var_name="legend", value_name="num_instances")

    #palette = sns.color_palette(, as_cmap=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(
        ax=ax,
        data=df,
        x="num_instances", 
        hue="legend",
        bins=bins, 
        palette="light:b", 
        edgecolor="black",
        alpha=0.7,
        log_scale=True,
        )
    plt.show()

def plot_histogram(all_traces, bins=100):
    
    row_maxes = torch.max(all_traces, dim=1).values  # shape: (num_rows,)
    row_mins = torch.min(all_traces, dim=1).values   # shape: (num_rows,)

    # Convert to numpy for plotting
    max_vals = row_maxes.numpy()
    min_vals = row_mins.numpy()

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram for max values
    axes[0].hist(max_vals, bins=bins, color='skyblue', edgecolor='black')
    axes[0].set_title('Histogram of Row-wise Max Values')
    axes[0].set_xlabel('Max Value')
    axes[0].set_ylabel('Frequency')

    # Histogram for min values
    axes[1].hist(min_vals, bins=bins, color='salmon', edgecolor='black')
    axes[1].set_title('Histogram of Row-wise Min Values')
    axes[1].set_xlabel('Min Value')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


    # Create a DataFrame for seaborn
    df = pd.DataFrame({
        'value': list(max_vals) + list(min_vals),
        'Legend': ['Positive amplitude'] * len(row_maxes) + ['Negative amplitude'] * len(row_mins)
    })


    # Plot using seaborn
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x='value', hue='Legend', bins=100, kde=False, palette="vlag", hue_order=['Negative amplitude', 'Positive amplitude'], edgecolor='black', alpha=1)
    #plt.title('Histogram of Row-wise Max and Min Values')
    plt.xlabel('Amplitude [mV]', fontsize=10)
    plt.ylabel('Frequency [-]', fontsize=10)
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame({
        'Positive amplitude [mV]': row_maxes,
        'Absolute value of negative amplitude [mV]': torch.abs(row_mins)
    })

    # Plot joint histogram
    plt.figure(figsize=(4, 4))
    #sns.jointplot(data=df, x='Positive amplitude [mV]', y='Absolute value of negative amplitude [mV]', kind='hist', bins=100, marginal_kws=dict(bins=50))
    sns.jointplot(data=df, x='Positive amplitude [mV]', y='Absolute value of negative amplitude [mV]',marker="+", s=100, marginal_kws=dict(bins=50, fill=False),)

    plt.xlabel('Positive amplitude [mV]', fontsize=10)
    plt.ylabel('Absolute value of negative amplitude [mV]', fontsize=10)
    plt.show()


# Mean and std computation for the purpose of Z-score normalization
def compute_mean_std_from_hdf(base_dir: str, max_files: int = 20, segment_ms: int = 100):
    trace_list = []
    files_loaded = 0

    for root, dirs, files in os.walk(base_dir):
        hdf_files = [f for f in files if f.endswith(".hdf")]
        for fname in hdf_files:
            if files_loaded >= max_files:
                break
            path = os.path.join(root, fname)
            try:
                traces, fs, metadata = read_hdf(path, return_fs=True, metadata_keys=["rov LAT", "end time", "utilized"])
                utilized, t_amp, t_last = metadata["utilized"], metadata["rov LAT"], metadata["end time"]
                traces = traces[utilized,:]
                t_amp = t_amp[utilized]
                t_last = t_last[utilized]
                indices = compute_amplitude_indices(t_amp, t_last, fs, traces.shape[1])
                traces = segment_signals(traces, indices, segment_ms, fs)
                traces = torch.tensor(traces, dtype=torch.float32)

                trace_list.append(traces)
                files_loaded += 1
            except Exception as e:
                print(f"Failed to read {path}: {e}")
        if files_loaded >= max_files:
            break

    if not trace_list:
        print("No valid HDF files found.")
        return None

    all_traces = torch.cat(trace_list, dim=0)  # (N_total, 2035)
    #all_traces = torch.tanh(all_traces / 5)
    mean = all_traces.mean()
    std = all_traces.std()
    min_val = all_traces.min()
    max_val = all_traces.max()

    row_maxes = torch.max(all_traces, dim=1).values  # shape: (num_rows,)
    row_mins = torch.min(all_traces, dim=1).values   # shape: (num_rows,)

    # Mean of row-wise max and min values
    mean_max = row_maxes.mean()
    mean_min = row_mins.mean()
    median_max = row_maxes.median()
    median_min = row_mins.median()
    pos_quantil_80= torch.quantile(row_maxes, 0.85)
    neg_quantil_20= torch.quantile(row_mins, 0.15)


    # plot_histogram(row_maxes, row_mins, bins = 100)

    print(f"Concatenated tensor shape: {all_traces.shape}")
    print(f"Mean: {mean.item():.4f}")
    print(f"Std: {std.item():.4f}")
    print(f"Min: {min_val.item():.4f}")
    print(f"Max: {max_val.item():.4f}")

    print(f"Mean of row-wise max: {mean_max.item():.4f}")
    print(f"Mean of row-wise min: {mean_min.item():.4f}")
    print(f"Median of row-wise max: {median_max.item():.4f}")
    print(f"Median of row-wise min: {median_min.item():.4f}")
    print(f"85th percentile of row-wise max: {pos_quantil_80.item():.4f}")
    print(f"15th percentile of row-wise min: {neg_quantil_20.item():.4f}")
    print(f"Number of files loaded: {files_loaded}")

    return all_traces


if __name__ == "__main__":
    # Example usage
    base_directory = "/media/guest/DataStorage/WaveMap/HDF5"  # Replace with your directory path
    df = scan_hdf_directories(base_directory)
    print(df.head())

    #plot_instances_histogram(df, bins=20)
    """ all_traces = compute_mean_std_from_hdf(base_directory, max_files=150, segment_ms=100)
    plot_histogram(all_traces, bins=100) """

    #print("Done")



