#%%
import numpy as np
from scipy.signal import correlate
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

#%%
TRACES_PATH = "/home/guest/lib/data/post_analysis/significant_traces.npy"
ATTENTIONS_PATH = "/home/guest/lib/data/post_analysis/significant_attentions.npy"

# Load the significant traces and attentions
signals = np.load(TRACES_PATH, allow_pickle=True)
attention_weights = np.load(ATTENTIONS_PATH, allow_pickle=True)

signals = signals[attention_weights > 0.015]  # Filter signals based on attention weights
print(f"Loaded {signals.shape[0]} signals after filtering by attention weights.")
#%%
""" def max_cross_correlation(sig1, sig2):
    corr = correlate(sig1, sig2, mode='full')
    corr /= (np.linalg.norm(sig1) * np.linalg.norm(sig2))
    return np.max(corr) """

def max_abs_cross_correlation(sig1, sig2):
    """Cross-correlation similarity, invariant to polarity (flip-insensitive)."""
    corr1 = correlate(sig1, sig2, mode='full')
    corr2 = correlate(sig1, -sig2, mode='full')

    norm_factor = np.linalg.norm(sig1) * np.linalg.norm(sig2)
    if norm_factor == 0:
        return 0.0  # handle edge case of zero signal

    corr1 /= norm_factor
    corr2 /= norm_factor

    return max(np.max(corr1), np.max(corr2))

N = signals.shape[0]
similarity_matrix = np.zeros((N, N))

for i in tqdm(range(N)):
    for j in range(i, N):
        sim = max_abs_cross_correlation(signals[i], signals[j])
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim

#%%
distance_matrix = 1 - similarity_matrix
np.fill_diagonal(distance_matrix, 0.0)
# Convert to condensed form (required by linkage)
condensed_dist = squareform(distance_matrix)

Z = linkage(condensed_dist, method='complete')  # or 'complete', 'ward'

# Plot dendrogram to choose cluster cut-off
plt.figure(figsize=(12, 5))
dendrogram(Z, no_labels=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Signal index")
plt.ylabel("Distance (1 - max cross-correlation)")
plt.show()
#%%
# Choose a threshold distance (e.g. 0.3)
labels = fcluster(Z, t=0.45, criterion='distance')
print(np.unique(labels, return_counts=True))
#%%
# --- Step 3: Apply DBSCAN using precomputed distances ---
dbscan = DBSCAN(eps=0.01, min_samples=10, metric='precomputed')
labels = dbscan.fit_predict(distance_matrix)
print(np.unique(labels, return_counts=True))

#%%
def plot_random_signals(signals, label):
    n = signals.shape[0]
    random_indices = np.random.choice(n, size=10, replace=False)

    # Select those 10 signals
    selected_signals = signals[random_indices]

    # Plot in a 5x2 grid
    fig, axes = plt.subplots(5, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig.suptitle(f"Random Signals from Cluster {label}", fontsize=16)

    for i in range(10):
        axes[i].plot(selected_signals[i])
        axes[i].set_title(f"Signal {random_indices[i]}")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

for label in np.unique(labels):
    print(f"Cluster {label} contains {np.sum(labels == label)} signals")
    signal_cluster = signals[labels == label]
    if signal_cluster.shape[0] > 10:
        print(f"Plotting random signals from cluster {label}")
        plot_random_signals(signal_cluster, label)
    else:
        print(f"Cluster {label} has only {signal_cluster.shape[0]} signals, skipping plotting.")
        continue