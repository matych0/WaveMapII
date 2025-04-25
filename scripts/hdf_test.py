import os
import h5py
import pandas as pd
import torch
import numpy as np
from typing import Union

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
                traces, fs = read_hdf(first_hdf_path, return_fs=True)
                shape = traces.shape
            except Exception as e:
                print(f"Failed to read {first_hdf_path}: {e}")
                shape = None
                fs = None

            rows.append({
                "subfolder": subfolder_name,
                "shape": shape,
                "fs": fs,
                "hdf_file_count": hdf_count
            })
    
    df = pd.DataFrame(rows)
    num_instances = [shape[0] for shape in df["shape"]]
    min_num_instances = min(num_instances)
    max_num_instances = max(num_instances)
    mean_num_instances = sum(num_instances) / len(num_instances)
    print(num_instances)
    print(f"Min: {min_num_instances}, Max: {max_num_instances}, Mean: {mean_num_instances}")
    return df

    # Save the DataFrame to a CSV file
    #df.to_csv("/media/guest/DataStorage/WaveMap/WaveMapEnsiteAnnotations/hdf_dataset_overview.csv", index=False)


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
                traces, fs, metadata = read_hdf(path, return_fs=True, metadata_keys=["rov LAT", "end time"])
                t_amp, t_last = metadata["rov LAT"], metadata["end time"]
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
    mean = all_traces.mean()
    std = all_traces.std()

    print(f"Concatenated tensor shape: {all_traces.shape}")
    print(f"Mean: {mean.item():.4f}")
    print(f"Std: {std.item():.4f}")

    return mean, std


if __name__ == "__main__":
    # Example usage
    base_directory = "/media/guest/DataStorage/WaveMap/HDF5"  # Replace with your directory path
    """ df = scan_hdf_directories(base_directory)
    print(df.head()) """

    mean, std = compute_mean_std_from_hdf(base_directory, max_files=150, segment_ms=100)
    print(f"Mean: {mean}, Std: {std}")



