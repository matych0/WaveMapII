#%%
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
pkl_path = "C:/Users/marti/Documents/Diplomka/results/post_analysis/df_clustering.pkl"

df = pd.read_pickle(pkl_path)
df["Attention Level"] = df["Attention Level"].apply(lambda x: "High" if x == "Q4" else "Low")
#df = df[df["Attention Level"] == "Q4"]

#%%
X = np.stack(df['feature_vector'].values)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(X)  # shape: (n_samples, 2)

df['UMAP1'] = embedding_2d[:, 0]
df['UMAP2'] = embedding_2d[:, 1]

#%%

sns.scatterplot(
    data=df,
    x='UMAP1', y='UMAP2',
    hue='Attention Level',    # or 'label', 'attention level', etc.
    palette='dark:crimson_r',
    s=10, alpha=0.8
)

plt.tight_layout()
plt.show()