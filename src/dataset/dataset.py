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
        metadata_keys: list = None,
    ):  

    with h5py.File(file_path, 'r') as hf:
        traces = hf["traces"][:]
        fs = hf["traces"].attrs["sampling_frequency"]    
        if metadata_keys:
            metadata = {key: hf["key"] for key in metadata_keys}
            return traces, fs, metadata
        else:
            return traces
        
def segmentation():
    # Example values (assuming they are read from a file and stored as byte strings)
    fs = 1000  # Sampling frequency in Hz
    T = 2  # Signal duration in seconds
    t_last = b'1708200000.500'  # Example Unix timestamp of last sample
    t_amp = b'1708199999.250'  # Example Unix timestamp of amplitude event

    # Convert bytes to float
    t_last = float(t_last.decode()) if isinstance(t_last, bytes) else float(t_last)
    t_amp = float(t_amp.decode()) if isinstance(t_amp, bytes) else float(t_amp)

    # Compute the number of samples
    N = int(T * fs)

    # Compute the first sample timestamp
    t_first = t_last - (N - 1) / fs

    # Compute the sample index
    k = round((t_amp - t_first) * fs)

    print(f"The amplitude occurs at sample index: {k}")
        

class HDFDataset(Dataset):
    def __init__(
            self,
            annotations_file: os.PathLike,
            data_dir: os.PathLike,
            train: bool = True,
            num_traces: int = None,
            transform = None,            
			startswith: str = "",
            readjustonce: bool = False,
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

        #control filepaths
        self.control_filepaths, self.control_maps = list(), list()
        for study_id in self.annotations["eid"].values:
            dir_name = glob.glob(os.path.join(self.data_dir, study_id, f"{startswith}*"), recursive=False)[0]
            file_fullpath = os.path.join(dir_name, os.path.basename(dir_name + ".hdf"))
            self.control_filepaths.append(file_fullpath)
            # read each file memory        
            if self.readjustonce:
                self.control_maps.append(read_hdf(file_fullpath))
        
        #case filepaths
        self.case_filepaths, self.case_maps = list(), list()
        for study_id in self.reccurence["eid"].values:
            dir_name = glob.glob(os.path.join(self.data_dir, study_id, f"{startswith}*"), recursive=False)[0]
            file_fullpath = os.path.join(dir_name, os.path.basename(dir_name + ".hdf"))
            self.case_filepaths.append(file_fullpath)
            # read each file memory        
            if self.readjustonce:
                self.case_maps.append(read_hdf(file_fullpath))
                    

    def __len__(self):
        return len(self.reccurence)

    def __getitem__(self, idx):
        time = self.reccurence.at[idx, 'days_to_event']
        index = np.searchsorted(self.time_array, time)
        control_idx = np.random.choice(range(index, self.annotations.shape[0]))
        
        control_time = self.annotations.at[control_idx, 'days_to_event']
        
        if self.readjustonce:
            case = self.case_maps[idx]
            control = self.control_maps[control_idx]
        else:
            case = read_hdf(self.case_filepaths[idx])
            control = read_hdf(self.control_filepaths[control_idx])

        if self.num_traces:
            indices = torch.randperm(case.shape[0])[:self.num_traces]
            case = case[indices]
            indices = torch.randperm(control.shape[0])[:self.num_traces]
            control = control[indices]

        if self.transform:
            case = self.transform(case)
            control = self.transform(control)
            
        case = torch.from_numpy(case)
        control = torch.from_numpy(control)
        
        return case.unsqueeze(0), control.unsqueeze(0)


if __name__ == "__main__":
    annotation_filepath = "C:/Users/matych/Desktop/SampleDataset/event_data.csv"
    dataset_folderpath = 'C:/Users/matych/Desktop/SampleDataset'

    training_data = HDFDataset(
        annotations_file=annotation_filepath,
        data_dir=dataset_folderpath,
        train=True,
        transform=None,            
        startswith="LA",
        readjustonce=True, 
        num_traces=4000           
    )
    
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    
    
    """ for i in range(20):"""
    case, control = next(iter(train_dataloader)) 
    
    print(case.shape)
    print(control.shape)