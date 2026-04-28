import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster

# 1. Load the data
# Ensure 'odfpy' is installed to read .ods files
file_path = '/home/matych/lib/data/WaveMap/loss_spreadsheets/loss_outliers.ods'
df = pd.read_excel(file_path, engine='odf')

csv_df = pd.read_csv('/home/matych/lib/data/WaveMap/HDF5/annotations_complete.csv')

# 2. Preprocessing
# Select only the loss columns for clustering
features = ['gnn_loss', 'egm_resnet_loss', 'patch_resnet_loss']
X = df[features]

# Standardizing is crucial if the loss scales differ significantly
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Hierarchical Clustering
# 'ward' minimizes the variance within clusters
linked = linkage(X_scaled, method='ward')

max_d = 10
clusters = fcluster(linked, max_d, criterion='distance')

# 2. Add the cluster IDs to your original dataframe
df['cluster_id'] = clusters

df = df.merge(
    csv_df[['eid', 'utilized']], 
    left_on='study_id', 
    right_on='eid', 
    how='left'
)

# 3. Save to a new .ods file
# Note: You may need the 'odfpy' library installed to write .ods
#df.to_excel('clustered_results.ods', index=False, engine='odf')

print(f"Clustering complete! Created {df['cluster_id'].nunique()} clusters.")

# 4. Visualization
plt.figure(figsize=(12, 7))
dendrogram(linked,
           orientation='top',
           labels=df['study_id'].values,
           distance_sort='descending',
           show_leaf_counts=True)

plt.title('Hierarchical Clustering of Data Inputs (Model Losses)')
plt.xlabel('Study ID')
plt.ylabel('Euclidean Distance (Ward)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()