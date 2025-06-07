import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns


def complete_ids(main_csv_path, mapping_csv_path, output_csv_path=None):
    # Load the data
    df_main = pd.read_csv(main_csv_path)
    df_mapping = pd.read_csv(mapping_csv_path)

    # Ensure the mapping DataFrame does not contain duplicates
    df_mapping = df_mapping.dropna(subset=['WaveMapID', 'EnSiteID']).drop_duplicates(subset='EnSiteID')

    # Merge to fill missing WaveMapID
    df_completed = df_main.merge(df_mapping, on='EnSiteID', how='left', suffixes=('', '_mapped'))

    # Fill in missing WaveMapID from mapping
    df_completed['WaveMapID'] = df_completed['WaveMapID'].fillna(df_completed['WaveMapID_mapped']).astype('int')

    # Drop the temporary mapped column
    df_completed = df_completed.drop(columns=['WaveMapID_mapped'])

    # Optionally save to a new CSV
    if output_csv_path:
        df_completed.to_csv(output_csv_path, index=False)

    return df_completed


def add_elapsed_days(completed_csv_path, elapsed_csv_path, output_csv_path=None):
    """
    Adds 'ElapsedDays' column to completed DataFrame based on 'WaveMapID' matching 'ID' from elapsed file.

    Parameters:
    - completed_csv_path (str): Path to the CSV with completed WaveMapIDs.
    - elapsed_csv_path (str): Path to the CSV containing 'ID' and 'ElapsedDays'.
    - output_csv_path (str, optional): If provided, saves the result to this path.

    Returns:
    - pd.DataFrame: Merged DataFrame with 'ElapsedDays' column added.
    """

    # Load both files
    df_completed = pd.read_csv(completed_csv_path)
    df_elapsed = pd.read_csv(elapsed_csv_path)

    # Make sure 'ID' and 'ElapsedDays' are in df_elapsed
    if not {'ID', 'ElapsedDays'}.issubset(df_elapsed.columns):
        raise ValueError("Elapsed days CSV must contain 'ID' and 'ElapsedDays' columns.")

    # Merge based on WaveMapID from df_completed and ID from df_elapsed
    df_merged = df_completed.merge(df_elapsed[['ID', 'ElapsedDays']],
                                   left_on='WaveMapID', right_on='ID', how='left')

    # Drop the redundant 'ID' column
    df_merged = df_merged.drop(columns=['ID'])

    # Create recurrence column: 1 if ElapsedDays is present, 0 if NaN
    df_merged['recurrence'] = df_merged['ElapsedDays'].notna().astype(int)

    # Fill NaN ElapsedDays with 365 and convert to integer
    df_merged['ElapsedDays'] = df_merged['ElapsedDays'].fillna(365).astype(int)

    # Save if path provided
    if output_csv_path:
        df_merged.to_csv(output_csv_path, index=False)

    return df_merged


def mark_exported(annotations_merged_csv_path, exported_ids_csv_path, output_csv_path=None):
    """
    Adds an 'exported' column to the DataFrame by merging with exported IDs.
    'exported' is 1 if 'EnSiteID' is in the exported list, else 0.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'EnSiteID' column.
    - exported_ids_csv_path (str): Path to CSV with exported 'EnSiteID' values.

    Returns:
    - pd.DataFrame: The DataFrame with an added 'exported' column.
    """
    df = pd.read_csv(annotations_merged_csv_path)
    # Load and prepare the exported IDs
    df_exported = pd.read_csv(exported_ids_csv_path)

    if 'EnSiteID' not in df_exported.columns:
        raise ValueError("Exported CSV must contain 'EnSiteID' column.")

    # Standardize to string and strip whitespace
    df_exported['EnSiteID'] = df_exported['EnSiteID'].astype(str).str.strip()

    # Add an indicator column before merging
    df_exported['exported'] = 1

    # Merge to assign the 'exported' status
    df_merged = df.merge(df_exported[['EnSiteID', 'exported', 'utilized']],
                         on='EnSiteID', how='left')

    # Fill NaNs in 'exported' with 0 (i.e., not exported)
    df_merged['exported'] = df_merged['exported'].fillna(0).astype(int)

    df_merged['utilized'] = df_merged['utilized'].fillna(0).astype(int)

    # Save if path provided
    if output_csv_path:
        df_merged.to_csv(output_csv_path, index=False)

    return df_merged


def assign_stratified_folds(annotation_csv_path, output_csv_path=None, n_splits=3, random_state=5032001):
    """
    Reads the annotation CSV, filters by 'exported' == 1, drops 'exported' column,
    and assigns stratified fold numbers (1-based) to a new 'fold' column.

    Parameters:
    - annotation_csv_path (str): Path to CSV with 'exported' and 'recurrence' columns.
    - n_splits (int): Number of folds (default is 3).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: Filtered and annotated DataFrame with 'fold' column added.
    """

    # Load and filter
    df = pd.read_csv(annotation_csv_path)
    df = df[df['exported'] == 1].drop(columns=['exported']).reset_index(drop=True)

    # Ensure 'recurrence' exists
    if 'recurrence' not in df.columns:
        raise ValueError("The CSV must contain a 'recurrence' column.")

    # Prepare StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize fold column
    df['fold'] = -1

    # Assign fold numbers (1-based)
    for fold_number, (_, val_idx) in enumerate(skf.split(df, df['recurrence'])):
        df.loc[val_idx, 'fold'] = fold_number

    df["training"] = df["fold"].apply(lambda x: 1 if (x == 1 or x == 2) else 0)

    # Save the DataFrame with folds if path provided
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)

    return df



def compute_wasserstein_distance_across_folds(df, column, n_splits):
    global_values = df[column].values
    dists = []

    for fold in range(n_splits):
        fold_values = df[df['fold'] == fold][column].values
        dist = wasserstein_distance(global_values, fold_values)
        dists.append(dist)

    return np.mean(dists)


def plot_utilized_histograms(df, column='utilized', fold_column='fold', n_splits=3, bins=30):
    """
    Plots histograms of the `column` for each fold.
    
    Parameters:
    - df: DataFrame with `column` and `fold_column`.
    - column: Name of the numeric variable to plot.
    - fold_column: Name of the column indicating fold assignment.
    - n_splits: Number of folds.
    - bins: Number of bins for the histogram.
    """
    plt.figure(figsize=(4, 16))

    for fold in range(n_splits + 1):
        plt.subplot(1, n_splits + 1, fold + 1)
        subset = df[df[fold_column] == fold][column]
        plt.hist(subset, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        plt.title(f'Fold {fold} (n={len(subset)})')
        plt.xlabel(column)
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()


def plot_utilized_histograms_seaborn(df, column='utilized', fold_column='fold', bins=30):
    """
    Uses Seaborn to plot histograms of `column` for each fold in separate subplots.
    """
    g = sns.FacetGrid(df, col=fold_column, col_wrap=1, sharex=True, sharey=True, height=3, aspect=1.5)
    g.map_dataframe(sns.histplot, x=column, bins=bins, kde=True, color="cornflowerblue", edgecolor="black")
    g.set_titles("Fold {col_name}")
    g.set_axis_labels("Number of electrograms [-]", "Frequency [-]")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


def assign_stratified_folds_optimized(annotation_csv_path, output_csv_path=None,
                                      n_splits=3, n_trials=500, random_state=5032001):
    """
    Optimizes stratified K-fold assignment by minimizing Wasserstein distance in 'utilized' column.
    """

    df_orig = pd.read_csv(annotation_csv_path)
    df_orig = df_orig[df_orig['exported'] == 1].drop(columns=['exported']).reset_index(drop=True)

    if 'recurrence' not in df_orig.columns or 'utilized' not in df_orig.columns:
        raise ValueError("The CSV must contain 'recurrence' and 'utilized' columns.")

    best_df = None
    best_wd = float('inf')

    rng = np.random.RandomState(random_state)

    for trial in range(n_trials):
        df = df_orig.copy()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng.randint(0, 1e9))
        df['fold'] = -1

        for fold_number, (_, val_idx) in enumerate(skf.split(df, df['recurrence'])):
            df.loc[val_idx, 'fold'] = fold_number

        wd_dist = compute_wasserstein_distance_across_folds(df, 'utilized', n_splits)

        if wd_dist < best_wd:
            best_wd = wd_dist
            best_df = df
            print(f"Trial {trial + 1}/{n_trials}: mean Wasserstein distance = {wd_dist:.4f}")
            #plot_utilized_histograms(best_df, column='utilized', fold_column='fold', n_splits=3, bins=30)
    plot_utilized_histograms_seaborn(best_df, column='utilized', fold_column='fold', bins=30)
    if output_csv_path:
        best_df.to_csv(output_csv_path, index=False)

    return best_df


def identify_non_paired_ids(annotations_merged_csv_path, hdf_exported_csv_path):

    annotations_df = pd.read_csv(annotations_merged_csv_path)
    hdf_df = pd.read_csv(hdf_exported_csv_path)

    # Specify the column containing IDs
    id_column = "EnSiteID"  # change this if your column is named differently

    # Get sets of unique IDs
    ann_ids = set(annotations_df[id_column])
    hdf_ids = set(hdf_df[id_column])

    # Symmetric difference: IDs in either file1 or file2 but not both
    exclusive_ids = hdf_ids - ann_ids

    print(f"Unique IDs not in annotation file: {exclusive_ids}")

    duplicated_ids = annotations_df['EnSiteID'][annotations_df['EnSiteID'].duplicated()].unique()
    print(f"{duplicated_ids} are duplicated in the annotations file.")


if __name__ == "__main__":
    annotations_merged = "/media/guest/DataStorage/WaveMap/WaveMapEnsiteAnnotations/annotations_merged.csv"
    exported_hdfs = "/media/guest/DataStorage/WaveMap/WaveMapEnsiteAnnotations/hdf_dataset_overview_overall.csv"
    annotations_export = "/media/guest/DataStorage/WaveMap/WaveMapEnsiteAnnotations/annotations_exported_complete.csv"    #annotations_export = "/media/guest/DataStorage/WaveMap/WaveMapEnsiteAnnotations/annotations_export.csv"
    annotations_complete = "/media/guest/DataStorage/WaveMap/WaveMapEnsiteAnnotations/annotations_complete.csv"
    
    #exported_df = mark_exported(annotations_merged, exported_hdfs, annotations_export)

    """ annotations_complete_df = assign_stratified_folds_optimized(annotations_export,
                                                               n_splits=3, 
                                                               n_trials=2000, 
                                                               random_state=5032001) """
    
    plot_utilized_histograms_seaborn(pd.read_csv(annotations_complete), column='utilized', fold_column='fold', bins=30)