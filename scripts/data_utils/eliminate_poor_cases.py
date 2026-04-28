import pandas as pd

# Load the main CSV (with 'eid' column)
df = pd.read_csv("/home/matych/lib/data/WaveMap/HDF5/annotations_complete.csv")

# Load the CSV with eids to remove (no header → set header=None)
remove_df = pd.read_csv("/home/matych/lib/data/WaveMap/loss_spreadsheets/cases_to_review.csv", header=None)

# Extract the list of eids to remove
remove_eids = remove_df[0]

# Filter: keep only rows where 'eid' is NOT in remove_eids
filtered_df = df[~df["eid"].isin(remove_eids)]

# Save the result
filtered_df.to_csv("/home/matych/lib/data/WaveMap/HDF5/filtered_annotations.csv", index=False)