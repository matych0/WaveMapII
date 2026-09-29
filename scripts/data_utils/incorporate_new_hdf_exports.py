import os
import shutil

source_root = r"/home/matych/lib/data/WaveMap/HDF_reexport"      # new dataset
target_root = r"/home/matych/lib/data/WaveMap/HDF5" # existing dataset

for study_id in os.listdir(source_root):
    source_study_path = os.path.join(source_root, study_id)
    
    if not os.path.isdir(source_study_path):
        continue  # skip non-folders
    
    target_study_path = os.path.join(target_root, study_id)

    if not os.path.exists(target_study_path):
        print(f"[INFO] Creating missing folder: {target_study_path}")
        os.makedirs(target_study_path)

    for file in os.listdir(source_study_path):
        if file.endswith(".hdf") or file.endswith(".h5"):
            src_file = os.path.join(source_study_path, file)
            dst_file = os.path.join(target_study_path, file)

            print(f"Copying: {src_file} -> {dst_file}")
            shutil.copy2(src_file, dst_file)  # preserves metadata