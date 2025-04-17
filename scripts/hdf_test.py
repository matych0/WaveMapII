import os
import h5py
import pandas as pd
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
    print(df)

    # Save the DataFrame to a CSV file
    #df.to_csv("/media/guest/DataStorage/WaveMap/WaveMapEnsiteAnnotations/hdf_dataset_overview.csv", index=False)


if __name__ == "__main__":
    # Example usage
    base_directory = "/media/guest/DataStorage/WaveMap/HDF5"  # Replace with your directory path
    scan_hdf_directories(base_directory)


