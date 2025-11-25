import os
import glob
import csv
import re

def clean_subfolder_name(parent_name, subfolder_name):
    if subfolder_name.startswith(parent_name):
        match = re.search(r"\d+$", subfolder_name)
        if match:
            return str(int(match.group(0)))  # Remove leading zeros
    return subfolder_name

def find_study_dws_folder(subfolder):
    # Recursively search all levels inside the subfolder
    for root, dirs, _ in os.walk(subfolder):
        for d in dirs:
            match = re.search(r"(study_dws.*)", d)
            if match:
                return match.group(1)
    return ""

def list_folders(base_path, output_csv):
    nested_folders = glob.glob(os.path.join(base_path, "_*"), recursive=False)

    data = []
    for i, parent_folder in enumerate(nested_folders, 1):
        if os.path.isdir(parent_folder):
            print(f"[{i}/{len(nested_folders)}] Scanning: {parent_folder}")
            subfolders = glob.glob(os.path.join(parent_folder, "*/"))
            parent_name = os.path.basename(parent_folder).lstrip("_")

            for subfolder in subfolders:
                subfolder_name = os.path.basename(os.path.normpath(subfolder))
                clean_name = clean_subfolder_name(parent_name, subfolder_name)
                study_dws_name = find_study_dws_folder(subfolder)
                data.append([parent_name, clean_name, study_dws_name])

    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Parent Folder", "Subfolder", "Study DWS Folder"])
        writer.writerows(data)

# Example usage
base_directory = "/media/guest/DataStorage/WaveMap/Raw"  # Change this to your target folder
output_file = "output.csv"
list_folders(base_directory, output_file)