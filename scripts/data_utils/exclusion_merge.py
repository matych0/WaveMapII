import pandas as pd

# Load your full dataset
df_full = pd.read_csv("/home/matych/lib/data/WaveMap/HDF5/annotations.csv")

# Load the exclusion list (only eid + Exclude column)
df_excl = pd.read_csv("/home/matych/lib/data/WaveMap/loss_spreadsheets/cases_reviewed_results.csv")

# Ensure Exclude is numeric (just in case)
df_excl["Exclude"] = df_excl["Exclude"].fillna(0)  #.astype(int)

# Merge (left join on eid)
df_merged = df_full.merge(df_excl[["eid", "Exclude"]], on="eid", how="left")

# Replace NaN (non-matching rows) with 0
df_merged["Exclude"] = df_merged["Exclude"].fillna("0")

df_merged["Exclude"] = df_merged["Exclude"].replace("zvazit", 1).astype(int)

df_merged = df_merged[df_merged["Exclude"] == 0].drop(columns=["Exclude"])

# Save result if needed
df_merged.to_csv("/home/matych/lib/data/WaveMap/HDF5/filtered_annotations_consider_excluded.csv", index=False)