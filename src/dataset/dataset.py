import os
import glob
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def read_hdf(
        file_path: os.PathLike,
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


def segmentation(segment_ms, traces, fs):
    """
    Segments traces into fixed-length segments.
    """
    central_sample = traces.shape[-1] // 2
    total_samples = traces.shape[1]
    segment_samples = np.round((segment_ms/1000)*fs)
    left_window = segment_samples//2
    right_window = segment_samples - left_window
    start, end = (int(max(central_sample - left_window, 0)), int(min(central_sample + right_window, total_samples)))
    return traces[:, start:end] 


def collect_filepaths_and_maps(data, data_dir, startswith, readjustonce, segment_ms):
    filepaths, maps, = list(), list()
    for study_id in data["eid"].values:
        dir_name = glob.glob(os.path.join(data_dir, study_id, f"{startswith}*"), recursive=False)[0]
        file_fullpath = os.path.join(dir_name, os.path.basename(dir_name + ".hdf"))
        filepaths.append(file_fullpath)
        if readjustonce:
            traces, fs, metadata = read_hdf(file_fullpath, return_fs=True, metadata_keys=["rov LAT", "end time"])
            if segment_ms:
                t_amp, t_last = metadata["rov LAT"], metadata["end time"]
                indices = compute_amplitude_indices(t_amp, t_last, fs, traces.shape[1])
                traces = segment_signals(traces, indices, segment_ms, fs)
                #traces = segmentation(segment_ms, traces, fs)
            maps.append(traces)
            
    if readjustonce:
        return maps
    else:
        return filepaths

class HDFDataset(Dataset):
    def __init__(
            self,
            annotations_file: os.PathLike,
            data_dir: os.PathLike,
            train: bool = True,
            num_traces: int = None,
            transform = None,            
			startswith: str = "",
            readjustonce: bool = True,
            segment_ms: int = None,            
            ):
        
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = pd.read_csv(annotations_file)
        # get training/validation studies only
        self.annotations = self.annotations[self.annotations["training"] == train]
        # sort by days to event
        self.annotations = self.annotations.sort_values(by='days_to_event')
        self.annotations.reset_index(drop=True, inplace=True)
        self.time_array = self.annotations['days_to_event']
        # get reccurence cases only
        self.reccurence = self.annotations[self.annotations['reccurence'] == 1]
        self.reccurence.reset_index(drop=True, inplace=True)
        
        self.readjustonce = readjustonce
        self.num_traces = num_traces
        self.segment_ms = segment_ms
                
        self.control_maps = collect_filepaths_and_maps(self.annotations, self.data_dir, startswith, readjustonce, segment_ms)
        self.case_maps = collect_filepaths_and_maps(self.reccurence, self.data_dir, startswith, readjustonce, segment_ms)
                    

    def __len__(self):
        return len(self.reccurence)

    def __getitem__(self, idx):
        time = self.reccurence.at[idx, 'days_to_event']
        index = np.searchsorted(self.time_array, time)
        control_idx = np.random.choice(range(index, self.annotations.shape[0]))
        
        control_time = self.annotations.at[control_idx, 'days_to_event']
        print(f"Case time: {time}, Control time: {control_time}")
        
        if self.readjustonce:
            case = self.case_maps[idx]
            control = self.control_maps[control_idx]
        else:
            case, case_fs, case_metadata = read_hdf(self.case_maps[idx], return_fs=True, metadata_keys=["rov LAT", "end time"])
            control, control_fs, control_metadata = read_hdf(self.control_maps[control_idx], return_fs=True, metadata_keys=["rov LAT", "end time"])
            if self.segment_ms:
                t_amp_case, t_last_case = case_metadata["rov LAT"], case_metadata["end time"]
                t_amp_control, t_last_control = control_metadata["rov LAT"], control_metadata["end time"]
                case_indices = compute_amplitude_indices(t_amp_case, t_last_case, case_fs, case.shape[1])
                control_indices = compute_amplitude_indices(t_amp_control, t_last_control, control_fs, control.shape[1])
                case = segment_signals(case, case_indices, self.segment_ms, case_fs)
                control = segment_signals(control, control_indices, self.segment_ms, control_fs)   
        
        np.random.shuffle(case)
        np.random.shuffle(control)
        
        if self.num_traces:
            case = case[:self.num_traces]
            control = control[:self.num_traces]

        if self.transform:
            case = self.transform(case)
            control = self.transform(control)
            
        case = torch.from_numpy(case)
        control = torch.from_numpy(control)
        
        return case, control


if __name__ == "__main__":
    """ annotation_filepath = "C:/Users/matych/Desktop/SampleDataset/event_data.csv"
    dataset_folderpath = 'C:/Users/matych/Desktop/SampleDataset' """
    
    # annotation_filepath = "C:/Users/matych/Research/SampleDataset/event_data.csv"
    # dataset_folderpath = "C:/Users/matych/Research/SampleDataset"

    annotation_filepath = "/home/guest/lib/data/WaveMapSampleHDF/event_data.csv"
    dataset_folderpath = "/home/guest/lib/data/WaveMapSampleHDF"

    training_data = HDFDataset(
        annotations_file=annotation_filepath,
        data_dir=dataset_folderpath,
        train=True,
        transform=None,            
        startswith="LA",
        readjustonce=True, 
        num_traces=None,
        segment_ms=101,           
    )
    
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    
    
    """ for i in range(20):"""
    case, control = next(iter(train_dataloader)) 
    
    print(f"Case shape: {case.shape}")
    
    print(f"Control shape: {control.shape}")
    