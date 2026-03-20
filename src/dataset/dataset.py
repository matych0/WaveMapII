import glob
import os

import h5py
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Dataset, Data
from torch_geometric.nn import radius_graph


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
    Computes the sample indices where LAT occur.

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

        segment = signals[i, start:end]

        # If the extracted segment is shorter than expected, pad with zeros
        if len(segment) < segment_samples:
            segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')

        segments[i] = segment

    return segments


def collect_filepaths(annotations_df, data_dir, startswith):
    """ Collects filepaths from the dir."""
    filepaths = list()
    for study_id in annotations_df["eid"].values:
        file_fullpath = glob.glob(os.path.join(data_dir, study_id, f"{startswith}*"), recursive=False)[0]
        filepaths.append(file_fullpath)
    return filepaths


def read_maps(filepaths, segment_ms, filter_utilized):
    """ Reads maps from the filepaths."""
    maps = list()
    for file_fullpath in filepaths:
        traces, fs, metadata = read_hdf(file_fullpath, return_fs=True, metadata_keys=["rov LAT", "end time", "utilized"])
        utilized, t_amp, t_last = metadata["utilized"], metadata["rov LAT"], metadata["end time"]
        if filter_utilized:
            traces = traces[utilized, :]
            t_amp = t_amp[utilized]
            t_last = t_last[utilized]
        if segment_ms:
            indices = compute_LAT_indices(t_amp, t_last, fs, traces.shape[1])
            traces = segment_signals(traces, indices, segment_ms, fs)
        maps.append(traces)
    return maps


def read_patches(filepaths, segment_ms, filter_utilized):
    """ Reads patches from the filepaths."""
    patches = list()
    for file_fullpath in filepaths:
        traces, fs, metadata = read_hdf(file_fullpath, return_fs=True, metadata_keys=["rov LAT", "end time", "utilized", "pt number"])
        utilized, t_amp, t_last, pt_number = metadata["utilized"], metadata["rov LAT"], metadata["end time"], metadata["pt number"]
        if filter_utilized:
            traces = traces[utilized, :]
            t_amp = t_amp[utilized]
            t_last = t_last[utilized]
            pt_number = pt_number[utilized]

        file_patches = list()

        for pt in np.unique(pt_number):
            pt_mask = pt_number == pt
            patch = traces[pt_mask, :]
            t_amp_pt = t_amp[pt_mask]
            t_last_pt = t_last[pt_mask]
            indices = compute_LAT_indices(t_amp_pt, t_last_pt, fs, patch.shape[1])
            avg_index = np.mean(indices).astype(int)
            #re-center indices around the average LAT index
            indices.fill(avg_index)
            patch = segment_signals(patch, indices, segment_ms, fs)
            patch = pad_channels(patch, target_ch=24)

            file_patches.append(patch)

        patches.append(np.stack(file_patches, axis=0))
    return patches


def collect_filepaths_and_maps(data, data_dir, startswith, readjustonce, segment_ms, filter_utilized):
    """ Collects filepaths and maps from the dir."""
    filepaths, maps = list(), list()
    for study_id in data["eid"].values:
        file_fullpath = glob.glob(os.path.join(data_dir, study_id, f"{startswith}*"), recursive=False)[0]
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
            maps.append(traces)
            
    if readjustonce:
        return maps
    else:
        return filepaths


def collect_filepaths_and_maps_inference(data, data_dir, startswith, readjustonce, segment_ms, filter_utilized):
    """ Collects filepaths and maps from the dir for inference purpose."""
    filepaths, maps, all_trace_indices, all_peak_to_peak = list(), list(), list(), list()
    for study_id in data["eid"].values:
        file_fullpath = glob.glob(os.path.join(data_dir, study_id, f"{startswith}*"), recursive=False)[0]
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


def pad_channels(patch, target_ch=24):
    ch, s = patch.shape
    pad_ch = target_ch - ch
    if pad_ch < 0:
        # print(f"Warning: patch has {ch} channels. Truncating to {target_ch} channels.")
        return patch[:target_ch, :]

        # raise ValueError("CH exceeds target number of channels")
    else:
        return np.pad(
            patch,
            pad_width=((0, pad_ch), (0, 0)),
            mode="constant",
            constant_values=0
        )
    

def random_node_sample(data, radius, max_nodes=500):

    N = data.x.shape[0]

    if N > max_nodes:

        idx = torch.randperm(N)[:max_nodes]

        data.x = data.x[idx]
        data.pos = data.pos[idx]

        edge_index = radius_graph(data.pos, r=radius)
        data.edge_index = edge_index

        start, end = edge_index
        edge_len = torch.norm(data.pos[start] - data.pos[end], dim=1, keepdim=True)
        edge_len = edge_len / radius
        data.edge_attr = edge_len

    return data


class HDFDataset(Dataset):
    """ Single-control training dataset."""
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
        self.np_rng = np.random.default_rng(random_seed) 

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
    """ Single-control validation dataset."""
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
    """ Validation dataset for inference and post-analysis, extracts more metadata."""
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
        
        traces_orig = traces.copy()
        traces_orig = torch.from_numpy(traces_orig)

        if self.transform:
            traces = self.transform(traces)

        traces = torch.from_numpy(traces)
        

        return duration, event, traces, traces_orig, filepath, trace_indices, peak_to_peak


class EGMDataset(Dataset):
    """ Multiple-control training and validation dataset."""
    def __init__(
            self,
            annotations_file: os.PathLike,
            data_dir: os.PathLike,
			startswith: str = "",  
            training: bool = True,
            fold: int = 0,
            readjustonce: bool = True,  
            filter_center: str = None,
            oversampling_factor: int = None,
            controls_time_gaussian_std: int = 0, 
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

        if filter_center:
            self.annotations = self.annotations[self.annotations['CenterID'] == filter_center]
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
        
        controls_mask = self.annotations['reccurence'] == 0
        
        if controls_time_gaussian_std > 0:
            noise = self.np_rng.normal(loc=0, scale=controls_time_gaussian_std, size=controls_mask.sum())
            noise = np.round(noise).astype(int)
            self.annotations.loc[controls_mask, 'days_to_event'] += noise

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


class EGMPatchDataset(Dataset):
    """ Multiple-control training and validation dataset with patching."""
    def __init__(
            self,
            annotations_file: os.PathLike,
            data_dir: os.PathLike,
			startswith: str = "",  
            training: bool = True,
            fold: int = 0,
            readjustonce: bool = True,  
            oversampling_factor: int = None,
            filter_center: str = None,
            controls_time_gaussian_std: int = 0, 
            num_traces: int = None,
            segment_ms: int = None,
            filter_utilized: bool = False, 
            transform = None,  
            random_seed: int = 3052001,
            shuffle_annotations: bool = False,
            ):
        

        self.transform = transform
        self.readjustonce = readjustonce
        self.num_traces = num_traces
        self.segment_ms = segment_ms
        self.filter_utilized = filter_utilized  
        self.training = training
        
        # set the random seed for reproducibility
        self.np_rng = np.random.default_rng(random_seed)

        # read the annotations file
        self.annotations = pd.read_csv(annotations_file)

        if filter_center:
            self.annotations = self.annotations[self.annotations['CenterID'] == filter_center]
        
        # get training/validation studies only
        if training == True:
            self.annotations = self.annotations[self.annotations["fold"] != fold]
            if shuffle_annotations:
                self.annotations.reset_index(drop=True, inplace=True)
                self.annotations[["reccurence", "days_to_event"]] = self.annotations[["reccurence", "days_to_event"]].sample(frac=1, random_state=random_seed).reset_index(drop=True)
        else:
            self.annotations = self.annotations[self.annotations["fold"] == fold]
    
        # Recurrence cases oversampling 
        if oversampling_factor:
            self.reccurence = self.annotations[self.annotations['reccurence'] == 1]
            self.oversampled = pd.concat([self.reccurence] * (oversampling_factor - 1), ignore_index=True)
            self.annotations = pd.concat([self.annotations, self.oversampled], ignore_index=True)
        self.annotations.reset_index(drop=True, inplace=True)
        
        if controls_time_gaussian_std > 0:
            controls_mask = self.annotations['reccurence'] == 0
            noise = self.np_rng.normal(loc=0, scale=controls_time_gaussian_std, size=controls_mask.sum())
            noise = np.round(noise).astype(int)
            self.annotations.loc[controls_mask, 'days_to_event'] += noise

        # read the HDF files or collect filepaths

        if readjustonce:
            filepaths = collect_filepaths(self.annotations, data_dir, startswith)
            self.patches = read_patches(filepaths, segment_ms, filter_utilized)

        else:
            self.filepaths = collect_filepaths(self.annotations, data_dir, startswith)


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        duration = self.annotations.at[idx, 'days_to_event']
        event = self.annotations.at[idx, 'reccurence']

        if self.readjustonce:
            traces = self.patches[idx]

        else:
            patches, fs, metadata = read_hdf(self.filepaths[idx], return_fs=True, metadata_keys=["rov LAT", "end time", "utilized", "pt number"])
            utilized, t_amp, t_last, pt_number = metadata["utilized"], metadata["rov LAT"], metadata["end time"], metadata["pt number"]

            if self.filter_utilized:
                patches = patches[utilized, :]
                t_amp = t_amp[utilized]
                t_last = t_last[utilized]
                pt_number = pt_number[utilized]

            traces = list()

            for pt in np.unique(pt_number):
                pt_mask = pt_number == pt
                patch = patches[pt_mask, :]
                t_amp_pt = t_amp[pt_mask]
                t_last_pt = t_last[pt_mask]
                indices = compute_LAT_indices(t_amp_pt, t_last_pt, fs, patch.shape[1])
                avg_index = np.mean(indices).astype(int)
                #re-center indices around the average LAT index
                indices.fill(avg_index)
                patch = segment_signals(patch, indices, self.segment_ms, fs)
                patch = pad_channels(patch, target_ch=24)

                traces.append(patch)

            traces = np.stack(traces, axis=0)

        
        if self.num_traces:
            if self.training:
                self.np_rng.shuffle(traces)

            traces = traces[:self.num_traces]

        if self.transform:
            traces = self.transform(traces)

        traces = torch.from_numpy(traces)
        duration = torch.tensor(duration, dtype=torch.float32)
        event = torch.tensor(event, dtype=torch.bool)
        
        return traces, duration, event
    

class AmplitudeDataset(Dataset):
    """ Dataset for training and validation on amplitude features only."""
    def __init__(
            self,
            annotations_file: os.PathLike,
            data_dir: os.PathLike,
            startswith: str = "",  
            training: bool = True,
            fold: int = 0,
            filter_utilized: bool = False, 
            num_traces: int = None,
            controls_time_gaussian_std: int = 0, 
            transform = None,
            random_seed: int = 3052001,
            shuffle_annotations: bool = False,
            **kwargs,
            ):
        
        self.transform = transform
        self.filter_utilized = filter_utilized
        self.num_traces = num_traces
        self.training = training

        # set the random seed for reproducibility
        self.np_rng = np.random.default_rng(random_seed)

        # read the annotations file
        self.annotations = pd.read_csv(annotations_file)

        if training == True:
            self.annotations = self.annotations[self.annotations["fold"] != fold]
            if shuffle_annotations:
                self.annotations.reset_index(drop=True, inplace=True)
                self.annotations[["reccurence", "days_to_event"]] = self.annotations[["reccurence", "days_to_event"]].sample(frac=1, random_state=random_seed).reset_index(drop=True)

        else: 
            self.annotations = self.annotations[self.annotations["fold"] == fold]

        self.annotations.reset_index(drop=True, inplace=True)
        
        if controls_time_gaussian_std > 0:
            controls_mask = self.annotations['reccurence'] == 0
            noise = self.np_rng.normal(loc=0, scale=controls_time_gaussian_std, size=controls_mask.sum())
            noise = np.round(noise).astype(int)
            self.annotations.loc[controls_mask, 'days_to_event'] += noise
        
        filepaths = collect_filepaths(self.annotations, data_dir, startswith)
        
        self.voltages = list()
        for file_fullpath in filepaths:
            _, _, metadata = read_hdf(file_fullpath, return_fs=False, metadata_keys=["peak2peak", "utilized", "roving x", "roving y", "roving z"])
            peak_to_peak, x, y, z = metadata["peak2peak"], metadata["roving x"], metadata["roving y"], metadata["roving z"]

            voltage = np.stack([peak_to_peak, x, y, z], axis=1)

            if self.filter_utilized:
                utilized = metadata["utilized"]
                voltage = voltage[utilized]
            self.voltages.append(voltage)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        duration = self.annotations.at[idx, 'days_to_event']
        event = self.annotations.at[idx, 'reccurence']
        voltage = self.voltages[idx]

        if self.num_traces:
            if self.training:
                self.np_rng.shuffle(voltage)

            voltage = voltage[:self.num_traces]

        if self.transform:
            voltage = self.transform(voltage)

        voltage = torch.from_numpy(voltage)
        duration = torch.tensor(duration, dtype=torch.float32)
        event = torch.tensor(event, dtype=torch.bool)

        return voltage, duration, event
    

class GraphFeatureDataset(Dataset):

    def __init__(
        self,
        annotations_file: os.PathLike,
        data_dir: os.PathLike,
        startswith: str = "",  
        training: bool = True,
        fold: int = 0,
        radius: float = 4.0,
        filter_utilized: bool = False, 
        num_traces: int = None,
        controls_time_gaussian_std: int = 0, 
        transform = None,
        random_seed: int = 3052001,
        **kwargs,
    ):

        super().__init__()

        self.radius = radius
        self.transform = transform
        self.filter_utilized = filter_utilized
        self.training = training
        self.num_traces = num_traces

        self.np_rng = np.random.default_rng(random_seed)

        self.annotations = pd.read_csv(annotations_file)

        if training:
            self.annotations = self.annotations[self.annotations["fold"] != fold]
        else:
            self.annotations = self.annotations[self.annotations["fold"] == fold]

        self.annotations.reset_index(drop=True, inplace=True)

        filepaths = collect_filepaths(self.annotations, data_dir, startswith)

        self.graphs = []

        for idx, file_fullpath in enumerate(filepaths):

            _, _, metadata = read_hdf(
                file_fullpath,
                return_fs=False,
                metadata_keys=["peak2peak", "utilized", "rov LAT", "ref LAT", "roving x", "roving y", "roving z"]
            )

            Vpp = metadata["peak2peak"]
            rov_LAT = metadata["rov LAT"]
            ref_LAT = metadata["ref LAT"]
            x = metadata["roving x"]
            y = metadata["roving y"]
            z = metadata["roving z"]

            coords = np.stack([x, y, z], axis=1)

            if self.filter_utilized:
                utilized = metadata["utilized"]
                coords = coords[utilized]
                Vpp = Vpp[utilized]
                rov_LAT = rov_LAT[utilized]
                ref_LAT = ref_LAT[utilized]

            # coord standardization
            coords_norm = np.copy(coords)
            coords_norm = (coords_norm - coords_norm.mean(axis=0)) / coords_norm.std(axis=0)

            # Vpp = (Vpp - Vpp.mean()) / Vpp.std()
            Vpp = np.tanh(Vpp / 5)
            #Vpp = (Vpp - Vpp.mean()) / Vpp.std()
            Vpp = np.stack([Vpp], axis=1)

            # LAT shifting to begin at 0
            rov_LAT = np.array([float(x.decode()) for x in rov_LAT])
            ref_LAT = np.array([float(x.decode()) for x in ref_LAT])
            delta_LAT = rov_LAT - ref_LAT
            delta_LAT = delta_LAT - delta_LAT.min()
            #delta_LAT = (delta_LAT - delta_LAT.mean()) / delta_LAT.std()
            delta_LAT = np.stack([delta_LAT], axis=1)

            # node features construction
            node_features = np.hstack([Vpp, delta_LAT, coords_norm]) # 

            # convert to tensors
            pos = torch.tensor(coords, dtype=torch.float)
            x_feat = torch.tensor(node_features, dtype=torch.float)

            # construct graph
            edge_index = radius_graph(pos, r=self.radius, loop=False)

            # compute edge lengths
            start, end = edge_index
            edge_len = torch.norm(pos[start] - pos[end], dim=1, keepdim=True)
            edge_len = edge_len / self.radius

            # label
            event = self.annotations.at[idx, "reccurence"]
            duration = self.annotations.at[idx, "days_to_event"]

            y_label = torch.tensor([[event, duration]], dtype=torch.float)

            graph = Data(
                x=x_feat,
                pos=pos,
                edge_index=edge_index,
                edge_attr=edge_len,
                y=y_label
            )

            self.graphs.append(graph)

    def len(self):
        return len(self.graphs)


    def get(self, idx):

        data = self.graphs[idx]

        if self.num_traces:
            #if self.training:
            data = random_node_sample(data, self.radius, max_nodes=self.num_traces)

        if self.transform:
            data = self.transform(data)

        return data
        


if __name__ == "__main__":
    annotations_file = "/home/matych/lib/data/WaveMap/HDF5/annotations_complete.csv"
    data_dir = "/home/matych/lib/data/WaveMap/HDF5/"
    
    """ dataset = EGMDataset(
        annotations_file=annotations_file,
        data_dir=data_dir,
        training=True,
        fold=0,
        readjustonce=True,
        controls_time_gaussian_std=5,
        segment_ms=100,
        num_traces=250,
        filter_utilized=False,
        transform=None,
        random_seed=3052001,
    ) """

    """ dataset = AmplitudeDataset(
        annotations_file=annotations_file,
        data_dir=data_dir,
        training=True,
        fold=0,
        filter_utilized=True,
        controls_time_gaussian_std=5,
        transform=None,
        random_seed=3052001,
        num_traces=1500,
    ) """
    
    """ print(f"Dataset size: {len(dataset)}")
    voltage, duration, event = dataset[2]
    print(f"Voltage shape: {voltage.shape}, Duration: {duration.shape}, Event: {event.shape}")
    """
    """  from .collate import collate_patches, collate_padding, collate_amplitudes
    from torch.utils.data import DataLoader
    

    # dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_padding)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_amplitudes)

    batch = next(iter(dataloader))
    print(f"Batched voltages shape: {batch[0].shape}, masks shape: {batch[1].shape}, durations shape: {batch[2].shape}, events shape: {batch[3].shape}") """

    from torch_geometric.loader import DataLoader
    
    dataset = GraphFeatureDataset(
        annotations_file=annotations_file,
        data_dir=data_dir,
        training=True,
        fold=0,
        radius=5.0,
        filter_utilized=True,
        num_traces=500,
        controls_time_gaussian_std=5,
        transform=None,
        random_seed=3052001,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )

    batch = next(iter(train_loader))
    print(batch)

    import matplotlib.pyplot as plt
    x = batch[0].x.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    sc = ax.scatter(
        x[:,-3],
        x[:,-2],
        x[:,-1],
        c=x[:,1],
        cmap='viridis',
        s=20
    )

    fig.colorbar(sc, ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()

    print("Done")
        

