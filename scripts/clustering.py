#%%
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import to_hex
import seaborn as sns
from scipy.spatial.distance import squareform
from scipy.signal import butter, filtfilt
from tslearn.barycenters import softdtw_barycenter
from sklearn.metrics import pairwise_distances_argmin_min

#%%
# Apply the bandpass filter
def bandpass_filter(data, lowcut=30.0, highcut=300.0, fs=2034.5, order=4):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    return filtfilt(b, a, data)

# Example: Apply filter to all signals in a DataFrame column
def filter_signals_in_dataframe(df, signal_column):
    df_filtered = df.copy()
    df_filtered[signal_column] = df_filtered[signal_column].apply(
        lambda sig: bandpass_filter(sig)
    )
    return df_filtered
#%%
# Step 1: Load the DataFrame from the pickle file
df = pd.read_pickle("C:/Users/marti/Documents/Diplomka/results/post_analysis/df_clustering.pkl")  # Replace with your actual file path
#df = df[df["Attention Level"] =="Q4"]  # Filter for high attention level
#%%
# Step 2: Stack all feature vectors into a 2D array
features = np.stack(df["feature_vector"].values)

# Step 3: Compute the distance matrix (Euclidean by default)
distance_matrix = pdist(features, metric='euclidean')

# Step 4: Perform hierarchical clustering
Z = linkage(distance_matrix, method="complete")  # 'ward' works well with Euclidean distances

#%%
# Step 5: Plot dendrogram
# Example: use a colormap (e.g., tab10, viridis, etc.)
cmap_orig = cm.rainbow(np.linspace(0, 1, 10))

hierarchy.set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap_orig])

plt.figure(figsize=(8, 4))
dendrogram(Z, color_threshold=18.5, no_labels=True)
plt.xlabel("High-attention electrograms")
plt.ylabel("Distance [-]")
plt.show()

# Step 6: Assign cluster labels (e.g., 7 clusters)
num_clusters = 3
df["cluster_id"] = fcluster(Z, num_clusters, criterion='maxclust')

# Optional: Inspect result
print(df[["feature_vector", "cluster_id"]].head())

#%%
orig_df = df.copy()  # Keep original DataFrame for later use
distance_matrix = squareform(distance_matrix)  # Now it's 2D!

#%%
plot_clusters = np.array([1, 2, 3])  # Specify clusters to plot
df = orig_df[orig_df["cluster_id"].isin(plot_clusters)]
cmap = cmap_orig[plot_clusters - 1]  # Adjust colormap to match clusters
df = filter_signals_in_dataframe(df, "traces")  # Apply bandpass filter to traces

N = 5  # number of signals per cluster
fs = 2035  # sampling frequency in Hz
cluster_col = "cluster_id"
signal_col = "traces"

""" # Get unique clusters
unique_clusters = sorted(df[cluster_col].unique())

# Sample signals first
sampled_traces = []
for cluster_id in unique_clusters:
    cluster_df = df[df[cluster_col] == cluster_id]
    sampled = cluster_df.sample(n=N, replace=len(cluster_df) < N, random_state=5032001)
    for _, row in sampled.iterrows():
        sampled_traces.append((cluster_id, row[signal_col])) """
        
N = 5  # signals per cluster
sampled_traces = []

# Reset index to keep alignment
df = df.reset_index(drop=True)
unique_clusters = sorted(df["cluster_id"].unique())
num_clusters = len(unique_clusters)

for cluster_id in unique_clusters:
    # Get the indices of rows in this cluster
    cluster_indices = df[df["cluster_id"] == cluster_id].index.to_numpy()

    # Submatrix of distances within this cluster
    sub_dist_matrix = distance_matrix[np.ix_(cluster_indices, cluster_indices)]

    # Compute distance to centroid (mean distance to all other points in cluster)
    dist_to_center = sub_dist_matrix.mean(axis=1)

    # Get indices of N most central samples
    top_indices = cluster_indices[np.argsort(dist_to_center)[:N]]

    for idx in top_indices:
        sampled_traces.append((cluster_id, df.at[idx, "traces"]))

# Compute y-axis limits from sampled traces
all_sampled_values = np.concatenate([trace for _, trace in sampled_traces])
v_min, v_max = -3, 4.2

# Organize sampled traces by cluster
cluster_to_traces = {cluster_id: [] for cluster_id in unique_clusters}
for cluster_id, trace in sampled_traces:
    cluster_to_traces[cluster_id].append(trace)

# Create subplots
fig, axes = plt.subplots(N, num_clusters, figsize=(num_clusters * 4, N * 2.5), sharex=False, sharey=True)

# Ensure axes is 2D
if num_clusters == 1:
    axes = np.expand_dims(axes, axis=1)
if N == 1:
    axes = np.expand_dims(axes, axis=0)

# Plot each trace
for col_idx, cluster_id in enumerate(unique_clusters):
    traces = cluster_to_traces[cluster_id]
    for row_idx, trace in enumerate(traces):
        ax = axes[row_idx, col_idx]
        
        time_ms = np.arange(len(trace)) / fs * 1000  # Convert to milliseconds
        
        cluster_color = mpl.colors.rgb2hex(cmap[col_idx][:3])  # match colormap order
        ax.plot(time_ms, trace, color=cluster_color, linewidth=1.5)
        ax.set_ylim(v_min, v_max)
        
        ax.grid(True, which='both', axis='both') 

        # Axis labeling
        if row_idx == N - 1:
            ax.set_xlabel("Time [ms]", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("Voltage [mV]", fontsize=11)

        # Reduce clutter
        if row_idx != N - 1:
            ax.set_xticklabels([])
            
        # Title for cluster
        if row_idx == 0:
            ax.set_title(f"Cluster {cluster_id}", fontsize=12)

# Global layout
plt.tight_layout()
plt.show()


#%%
df_melted = orig_df[["peak_to_peak", "extrema_count", "entropy", "dominant_frequency", "cluster_id"]].melt(id_vars='cluster_id', var_name='parameter', value_name='value')

y_labels = {
    'peak_to_peak': 'Peak to Peak Voltage [mV]',
    'extrema_count': 'Number of Local Extrema',
    'entropy': 'Sample Entropy',
    'dominant_frequency': 'Dominant Frequency [Hz]'
}


g = sns.catplot(
    data=df_melted,
    y='value',
    hue='cluster_id',
    kind='box',
    col='parameter',
    sharey=False,
    col_wrap=4,
    palette=cmap_orig,
    height=6, 
    aspect=0.5,
    showfliers=False
)

for ax in g.axes.flatten():
    for line in ax.lines:
        # Median lines in Seaborn boxplots are usually at index 4, 11, 18, ... etc.
        # So you can color all of them, or use pattern if you're confident of structure
        if line.get_linestyle() == '-':  # most median lines are solid
            line.set_color('black')      # change to desired color
            line.set_linewidth(1.5)        # optional: make it thicker

for ax, parameter in zip(g.axes.flat, df_melted['parameter'].unique()):
    ax.set_ylabel(y_labels.get(parameter, 'Value'))
    ax.set_title("")
    
g._legend.remove()
g.add_legend(title="Cluster", bbox_to_anchor=(0.5, 0.95), loc='center', ncol=7)

#plt.savefig('boxplot_all_parameters_three_cat.svg', format='svg')
plt.tight_layout()
plt.show()

#%%
barycenter_df = orig_df[orig_df["cluster_id"] != 5]
cluster_ids = sorted(barycenter_df['cluster_id'].unique())
n_clusters = len(cluster_ids)
vmin, vmax = -3, 4.2  # y-axis limits
fs = 2035  # Sampling rate

# Create colormap
cmap_orig = cm.rainbow(np.linspace(0, 1, 10))
colors = np.delete(cmap_orig[:n_clusters+1], 4, axis=0)  # remove cluster 5 color

# Determine subplot layout
n_cols = 2
n_rows = int(np.ceil(n_clusters / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), sharex=True, sharey=True)
axes = axes.flatten()  # to index with a single loop

# Plot each cluster
for idx, cluster_id in enumerate(cluster_ids):
    ax = axes[idx]
    group = barycenter_df[barycenter_df['cluster_id'] == cluster_id]

    signals = np.stack(group["traces"].values)
    signals = signals[..., np.newaxis]
    
    sz = signals.shape[1]
    time_ms = np.arange(sz) * (1000 / fs)

    barycenter = softdtw_barycenter(signals, gamma=10)

    for signal in signals:
        ax.plot(time_ms, signal.squeeze(), color='gray', alpha=0.3)

    ax.plot(time_ms, barycenter.squeeze(), color=colors[idx], linewidth=2, label='Barycenter')

    ax.set_ylim(vmin, vmax)
    ax.set_title(f"Cluster {cluster_id}")
    ax.set_ylabel("Voltage [mV]")
    ax.legend(loc='upper right')

# Hide unused subplots
for i in range(n_clusters, len(axes)):
    fig.delaxes(axes[i])

# Label bottom row
for ax in axes[-n_cols:]:
    ax.set_xlabel("Time [ms]")

plt.tight_layout()
plt.show()

#%%
k_range = range(1, 25)

# Store total within-cluster variance (inertia-like)
wcss = []  # within-cluster sum of squares

for k in k_range:
    labels = fcluster(Z, k, criterion='maxclust')
    
    # Compute total within-cluster sum of squares (like KMeans inertia)
    centroids = np.array([features[labels == i].mean(axis=0) for i in range(1, k+1)])
    closest, dists = pairwise_distances_argmin_min(features, centroids)
    wcss.append((dists**2).sum())

# Plot elbow
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.grid(True)
plt.show()