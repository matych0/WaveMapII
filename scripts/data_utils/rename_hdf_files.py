import os
import glob

# Root directory containing all study folders
data_dir = r"/home/matych/lib/data/WaveMap/HDF5"

# Iterate through all study folders
for study_id in os.listdir(data_dir):

    study_path = os.path.join(data_dir, study_id)

    # Skip non-directories
    if not os.path.isdir(study_path):
        continue

    # Get all files in the folder
    all_files = [
        f for f in os.listdir(study_path)
        if os.path.isfile(os.path.join(study_path, f))
    ]

    # Proceed only if there is exactly ONE file in the folder
    if len(all_files) == 1:

        filename = all_files[0]

        # Rename only if the file starts with LA_WMP2
        if filename.startswith("LA_WMP2"):

            old_path = os.path.join(study_path, filename)

            # Replace only the prefix
            new_filename = filename.replace(
                "LA_WMP2",
                "LA_WMP2_SR",
                1
            )

            new_path = os.path.join(study_path, new_filename)

            # Rename file
            os.rename(old_path, new_path)

            print(f"Renamed:\n  {filename}\n  -> {new_filename}\n")