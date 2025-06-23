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
          
    
def compute_LAT_indices(t_LAT, t_last, fs, num_samples):
    """
    Computes the sample indices where LAT events occur.

    Parameters:
    t_amp (np.ndarray): Array of Unix timestamps when amplitude events occur.
    t_last (np.ndarray): Array of Unix timestamps of the last sample.
    fs (float): Sampling frequency in Hz.
    num_samples (int): Number of samples in the signal.

    Returns:
    np.ndarray: Array of sample indices for each amplitude event.
    """
    # Ensure inputs are numpy arrays and convert byte strings if necessary
    t_LAT = np.array([float(x.decode()) if isinstance(x, bytes) else float(x) for x in t_LAT])
    t_last = np.array([float(x.decode()) if isinstance(x, bytes) else float(x) for x in t_last])

    # Compute the time of the first sample for each signal
    t_first = t_last - (num_samples - 1) / fs

    # Compute sample indices (rounded to nearest integer)
    sample_indices = np.round((t_LAT - t_first) * fs).astype(int)

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


def collect_filepaths_and_maps(data, data_dir, startswith, readjustonce, segment_ms, filter_utilized):
    filepaths, maps, = list(), list()
    for study_id in data["eid"].values:
        file_fullpath = glob.glob(os.path.join(data_dir, study_id, f"{startswith}*"), recursive=False)[0]
        #file_fullpath = os.path.join(file_fullpath, os.path.basename(file_fullpath + ".hdf"))
        filepaths.append(file_fullpath)
        if readjustonce:
            traces, fs, metadata = read_hdf(file_fullpath, return_fs=True, metadata_keys=["rov LAT", "end time", "utilized"])
            utilized, t_amp, t_last = metadata["utilized"], metadata["rov LAT"], metadata["end time"]
            if filter_utilized:
                traces = traces[utilized, :]
                t_amp = t_amp[utilized]
                t_last = t_last[utilized]
            if segment_ms:
                indices = compute_LAT_indices(t_amp, t_last, fs, traces.shape[1])
                traces = segment_signals(traces, indices, segment_ms, fs)
                #traces = segmentation(segment_ms, traces, fs)
            maps.append(traces)
            
    if readjustonce:
        return maps
    else:
        return filepaths


def collect_filepaths_and_maps_inference(data, data_dir, startswith, readjustonce, segment_ms, filter_utilized):
    filepaths, maps, all_trace_indices, all_peak_to_peak = list(), list(), list(), list()
    for study_id in data["eid"].values:
        file_fullpath = glob.glob(os.path.join(data_dir, study_id, f"{startswith}*"), recursive=False)[0]
        #file_fullpath = os.path.join(file_fullpath, os.path.basename(file_fullpath + ".hdf"))
        filepaths.append(file_fullpath)
        if readjustonce:
            traces, fs, metadata = read_hdf(file_fullpath, return_fs=True, metadata_keys=["rov LAT", "end time", "utilized", "peak2peak"])
            utilized, t_amp, t_last, peak_to_peak = metadata["utilized"], metadata["rov LAT"], metadata["end time"], metadata["peak2peak"]
            trace_indices = np.arange(0, len(traces), dtype=int)
            if filter_utilized:
                traces = traces[utilized, :]
                t_amp = t_amp[utilized]
                t_last = t_last[utilized]
                trace_indices = trace_indices[utilized]
                peak_to_peak = peak_to_peak[utilized]
            if segment_ms:
                indices = compute_LAT_indices(t_amp, t_last, fs, traces.shape[1])
                traces = segment_signals(traces, indices, segment_ms, fs)
                #traces = segmentation(segment_ms, traces, fs)
            maps.append(traces)

        all_trace_indices.append(trace_indices)
        all_peak_to_peak.append(peak_to_peak)

    return filepaths, maps, all_trace_indices, all_peak_to_peak



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
            filter_utilized: bool = False, 
            oversampling_factor: int = None,
            random_seed: int = 3052001,
            cross_val_fold: int = None,
            num_controls: int = 1,        
            ):
        
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = pd.read_csv(annotations_file)
        # get training/validation studies only
        if train == True:
            if cross_val_fold is None:
                self.annotations = self.annotations[self.annotations["training"] == True]
            else:
                self.annotations = self.annotations[self.annotations["fold"] != cross_val_fold]
        else:
            if cross_val_fold is None:
                self.annotations = self.annotations[self.annotations["training"] == False]
            else:
                self.annotations = self.annotations[self.annotations["fold"] == cross_val_fold]
    
        # get reccurence cases only
        self.reccurence = self.annotations[self.annotations['reccurence'] == 1]
        self.reccurence.reset_index(drop=True, inplace=True)
        self.np_rng = np.random.default_rng(random_seed) # NumPy random generator

        # Recurrence cases oversampling 
        if oversampling_factor:
            self.oversampled = pd.concat([self.reccurence] * (oversampling_factor - 1), ignore_index=True)
            self.annotations = pd.concat([self.annotations, self.oversampled], ignore_index=True)

        # sort by days to event
        self.annotations = self.annotations.sort_values(by='days_to_event')
        self.annotations.reset_index(drop=True, inplace=True)
        self.time_array = self.annotations['days_to_event']

        
        
        self.readjustonce = readjustonce
        self.num_traces = num_traces
        self.segment_ms = segment_ms
        self.filter_utilized = filter_utilized
                
        self.control_maps = collect_filepaths_and_maps(self.annotations, self.data_dir, startswith, readjustonce, segment_ms, filter_utilized)
        self.case_maps = collect_filepaths_and_maps(self.reccurence, self.data_dir, startswith, readjustonce, segment_ms, filter_utilized)
                    

    def __len__(self):
        return len(self.reccurence)

    def __getitem__(self, idx):
        time = self.reccurence.at[idx, 'days_to_event']
        index = np.searchsorted(self.time_array, time)
        control_idx = self.np_rng.choice(range(index, self.annotations.shape[0]))
        
        control_time = self.annotations.at[control_idx, 'days_to_event']
        # print(f"Case time: {time}, Control time: {control_time}")
        
        if self.readjustonce:
            case = self.case_maps[idx]
            control = self.control_maps[control_idx]

        else:
            case, case_fs, case_metadata = read_hdf(self.case_maps[idx], return_fs=True, metadata_keys=["rov LAT", "end time", "utilized"])
            control, control_fs, control_metadata = read_hdf(self.control_maps[control_idx], return_fs=True, metadata_keys=["rov LAT", "end time", "utilized"])
            utilized_case, t_amp_case, t_last_case = case_metadata["utilized"], case_metadata["rov LAT"], case_metadata["end time"]
            utilized_control, t_amp_control, t_last_control = control_metadata["utilized"], control_metadata["rov LAT"], control_metadata["end time"]
            
            if self.filter_utilized:
                case = case[utilized_case, :]
                control = control[utilized_control, :]
                t_amp_case = t_amp_case[utilized_case]
                t_last_case = t_last_case[utilized_case]
                t_amp_control = t_amp_control[utilized_control]
                t_last_control = t_last_control[utilized_control]

            if self.segment_ms:
                case_indices = compute_LAT_indices(t_amp_case, t_last_case, case_fs, case.shape[1])
                control_indices = compute_LAT_indices(t_amp_control, t_last_control, control_fs, control.shape[1])
                case = segment_signals(case, case_indices, self.segment_ms, case_fs)
                control = segment_signals(control, control_indices, self.segment_ms, control_fs)   
        
        if self.num_traces:
            np.random.shuffle(case)
            np.random.shuffle(control)
            case = case[:self.num_traces]
            control = control[:self.num_traces]

        if self.transform:
            case = self.transform(case)
            control = self.transform(control)
            
        case = torch.from_numpy(case)
        control = torch.from_numpy(control)
        
        return case, control
    

class ValidationDataset(Dataset):
    def __init__(
            self,
            annotations_file: os.PathLike,
            data_dir: os.PathLike,
            eval_data: bool = True,
            transform = None,            
            startswith: str = "",
            readjustonce: bool = True,
            segment_ms: int = None,
            filter_utilized: bool = False,
            cross_val_fold: int = None,            
            ):
        
        self.data_dir = data_dir
        self.transform = transform
        self.annotations = pd.read_csv(annotations_file)
        # get training/validation studies only
        if eval_data == True:
            if cross_val_fold is None:
                self.annotations = self.annotations[self.annotations["training"] == False]
            else:
                self.annotations = self.annotations[self.annotations["fold"] == cross_val_fold]
        else:
            if cross_val_fold is None:
                self.annotations = self.annotations[self.annotations["training"] == True]
            else:
                self.annotations = self.annotations[self.annotations["fold"] != cross_val_fold]

        self.annotations.reset_index(drop=True, inplace=True)
        
        self.readjustonce = readjustonce
        self.segment_ms = segment_ms
        self.filter_utilized = filter_utilized
        
        self.maps = collect_filepaths_and_maps(self.annotations, self.data_dir, startswith, readjustonce, segment_ms, filter_utilized)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        duration = self.annotations.at[idx, 'days_to_event']
        event = self.annotations.at[idx, 'reccurence']

        if self.readjustonce:
            traces = self.maps[idx]
                
        else:
            traces, fs, metadata = read_hdf(self.maps[idx], return_fs=True, metadata_keys=["rov LAT", "end time", "utilized"])
            utilized, t_amp, t_last = metadata["utilized"], metadata["rov LAT"], metadata["end time"]
            if self.filter_utilized:
                traces = traces[utilized, :]
                t_amp = t_amp[utilized]
                t_last = t_last[utilized]
            if self.segment_ms:
                case_indices = compute_LAT_indices(t_amp, t_last, fs, traces.shape[1])
                traces = segment_signals(traces, case_indices, self.segment_ms, fs)
                
        if self.transform:
            traces = self.transform(traces)
            
        traces = torch.from_numpy(traces)

        return duration, event, traces
    

class ValidationDatasetInference(Dataset):
    def __init__(
            self,
            annotations_file: os.PathLike,
            data_dir: os.PathLike,
            eval_data: bool = True,
            transform = None,            
            startswith: str = "",
            readjustonce: bool = True,
            segment_ms: int = None,
            filter_utilized: bool = False,
            cross_val_fold: int = None,
            ):

        self.data_dir = data_dir
        self.transform = transform
        self.annotations = pd.read_csv(annotations_file)
        # get training/validation studies only
        if eval_data == True:
            if cross_val_fold is None:
                self.annotations = self.annotations[self.annotations["training"] == False]
            else:
                self.annotations = self.annotations[self.annotations["fold"] == cross_val_fold]
        else:
            if cross_val_fold is None:
                self.annotations = self.annotations[self.annotations["training"] == True]
            else:
                self.annotations = self.annotations[self.annotations["fold"] != cross_val_fold]

        self.annotations.reset_index(drop=True, inplace=True)
        
        self.readjustonce = readjustonce
        self.segment_ms = segment_ms
        self.filter_utilized = filter_utilized

        self.filepaths, self.maps, self.trace_indices, self.peak_to_peak = collect_filepaths_and_maps_inference(self.annotations, self.data_dir, startswith, readjustonce, segment_ms, filter_utilized)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        duration = self.annotations.at[idx, 'days_to_event']
        event = self.annotations.at[idx, 'reccurence']
        filepath = self.filepaths[idx]
        trace_indices = self.trace_indices[idx]
        peak_to_peak = self.peak_to_peak[idx]

        if self.readjustonce:
            traces = self.maps[idx]

        else:
            traces, fs, metadata = read_hdf(self.maps[idx], return_fs=True, metadata_keys=["rov LAT", "end time", "utilized"])
            utilized, t_amp, t_last = metadata["utilized"], metadata["rov LAT"], metadata["end time"]
            trace_indices = np.arange(0, len(traces), dtype=int)
            if self.filter_utilized:
                traces = traces[utilized, :]
                t_amp = t_amp[utilized]
                t_last = t_last[utilized]
                trace_indices = trace_indices[utilized]
            if self.segment_ms:
                case_indices = compute_LAT_indices(t_amp, t_last, fs, traces.shape[1])
                traces = segment_signals(traces, case_indices, self.segment_ms, fs)
                
        if self.transform:
            traces = self.transform(traces)

        traces = torch.from_numpy(traces)

        return duration, event, traces, filepath, trace_indices, peak_to_peak


class EGMDataset(Dataset):
    def __init__(
            self,
            annotations_file: os.PathLike,
            data_dir: os.PathLike,
			startswith: str = "",  
            training: bool = True,
            fold: int = 0,
            readjustonce: bool = True,  
            oversampling_factor: int = None,    
            controls_time_shift: int = 0,    
            num_traces: int = None,
            segment_ms: int = None,
            filter_utilized: bool = False, 
            transform = None,  
            random_seed: int = 3052001,   
            ):
        

        self.transform = transform
        self.readjustonce = readjustonce
        self.num_traces = num_traces
        self.segment_ms = segment_ms
        self.filter_utilized = filter_utilized  
        
        # set the random seed for reproducibility
        self.np_rng = np.random.default_rng(random_seed)

        # read the annotations file
        self.annotations = pd.read_csv(annotations_file)
        # get training/validation studies only
        if training == True:
            self.annotations = self.annotations[self.annotations["fold"] != fold]
        else:
            self.annotations = self.annotations[self.annotations["fold"] == fold]
    
        # Recurrence cases oversampling 
        if oversampling_factor:
            self.reccurence = self.annotations[self.annotations['reccurence'] == 1]
            self.oversampled = pd.concat([self.reccurence] * (oversampling_factor - 1), ignore_index=True)
            self.annotations = pd.concat([self.annotations, self.oversampled], ignore_index=True)
        self.annotations.reset_index(drop=True, inplace=True)

        if controls_time_shift > 0:
            # Shift the controls by a random amount of time
            self.annotations['days_to_event'] += controls_time_shift

        # read the HDF files or collect filepaths
        self.maps = collect_filepaths_and_maps(self.annotations, data_dir, startswith, readjustonce, segment_ms, filter_utilized)
                    

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        duration = self.annotations.at[idx, 'days_to_event']
        event = self.annotations.at[idx, 'reccurence']

        if self.readjustonce:
            traces = self.maps[idx]

        else:
            traces, fs, metadata = read_hdf(self.maps[idx], return_fs=True, metadata_keys=["rov LAT", "end time", "utilized"])
            utilized, t_amp, t_last = metadata["utilized"], metadata["rov LAT"], metadata["end time"]
            
            if self.filter_utilized:
                traces = traces[utilized, :]
                t_amp = t_amp[utilized]
                t_last = t_last[utilized]

            if self.segment_ms:
                indices = compute_LAT_indices(t_amp, t_last, fs, traces.shape[1])
                traces = segment_signals(traces, indices, self.segment_ms, fs)
        
        if self.num_traces:
            self.np_rng.shuffle(traces)
            traces = traces[:self.num_traces]

        if self.transform:
            traces = self.transform(traces)

        traces = torch.from_numpy(traces)
        duration = torch.tensor(duration, dtype=torch.float32)
        event = torch.tensor(event, dtype=torch.bool)
        
        return traces, duration, event


if __name__ == "__main__":
    from collate import collate_padding, collate_validation, collate_padding_two_tensors, collate_padding_merged

    annotation_filepath = "/media/guest/DataStorage/WaveMap/HDF5/annotations_complete.csv"
    dataset_folderpath = "/media/guest/DataStorage/WaveMap/HDF5"

    training_data = EGMDataset(
        annotations_file=annotation_filepath,
        data_dir=dataset_folderpath,
        train=True,
        cross_val_fold=0,
        transform=None,            
        startswith="LA",
        readjustonce=False, 
        num_traces=None,
        segment_ms=100, 
        filter_utilized=True,
        oversampling_factor=4,          
    )
    
    train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True, collate_fn=collate_padding_merged)
    
    # for i in range(20):
    traces, mask = next(iter(train_dataloader)) 
    
    print(f"Traces shape: {traces.shape}")
    

    """ validation_data = ValidationDataset(
        annotations_file=annotation_filepath,
        data_dir=dataset_folderpath,
        transform=None,            
        startswith="LA",
        readjustonce=False, 
        segment_ms=100,
        filter_utilized=True,  
        eval_data=False, 
        #cross_val_fold=3,        
    )

    validation_dataloader = DataLoader(validation_data, batch_size=100, shuffle=False, collate_fn=collate_validation)

    duration, event, traces, traces_masks = next(iter(validation_dataloader))
    print(f"Durations: {duration}")
    print(f"Events: {event}")
    print(f"Traces shape: {traces.shape}") """



    