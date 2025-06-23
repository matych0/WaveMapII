#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
csv_path = "C:/Users/marti/Documents/Diplomka/data/inference_results.csv"

df = pd.read_csv(csv_path)

#%%
df['Attention Level'] = ['High (>0.01)' if x >= 0.01 else 'Low (<=0.01)' for x in df['attention']]
#%%
def label_attention(x):
    if x >= 0.01:
        return 'High (≥0.01)'
    elif x > 0.0001:
        return 'Medium (0.0001-0.01)'
    else:
        return 'Low (≤0.0001)'
    
df['Attention Level'] = df['attention'].apply(label_attention)
    
#%%
df_melted = df[["peak_to_peak", "extrema_count", "entropy", "dominant_frequency", "Attention Level"]].melt(id_vars='Attention Level', var_name='parameter', value_name='value')

y_labels = {
    'peak_to_peak': 'Peak to Peak Voltage [mV]',
    'extrema_count': 'Number of Local Extrema',
    'entropy': 'Sample Entropy',
    'dominant_frequency': 'Dominant Frequency [Hz]'
}


g = sns.catplot(
    data=df_melted,
    y='value',
    hue='Attention Level',
    kind='box',
    col='parameter',
    sharey=False,
    col_wrap=4,
    palette="vlag",
    height=6, 
    aspect=0.5
)

for ax, parameter in zip(g.axes.flat, df_melted['parameter'].unique()):
    ax.set_ylabel(y_labels.get(parameter, 'Value'))
    ax.set_title("")
    
plt.savefig('boxplot_all_parameters_three_cat.svg', format='svg')

plt.show()

#%%
plt.figure(figsize=(4,7))
sns.boxplot(x='Attention Level', y='peak_to_peak', data=df, palette="vlag")
plt.ylabel('Peak to Peak Voltage [mV]')
plt.xlabel('Attention Level')
plt.tight_layout()
plt.savefig('peak_to_peak.svg', format='svg')
plt.show()


plt.figure(figsize=(4,7))
sns.boxplot(x='Attention Level', y='extrema_count', data=df, palette="vlag")
plt.ylabel('Number of Local Extrema')
plt.xlabel('Attention Level')
plt.tight_layout()
plt.savefig('extrema_count.svg', format='svg')
plt.show()

plt.figure(figsize=(4,7))
sns.boxplot(x='Attention Level', y='entropy', data=df, palette="vlag")
plt.ylabel('Sample Entropy')
plt.xlabel('Attention Level')
plt.tight_layout()
plt.savefig('entropy.svg', format='svg')
plt.show()

plt.figure(figsize=(4,7))
sns.boxplot(x='Attention Level', y='dominant_frequency', data=df, palette="vlag")
plt.ylabel('Dominant Frequency [Hz]')
plt.xlabel('Attention Level')
plt.tight_layout()
plt.savefig('dominant_frequency.svg', format='svg')
plt.show()

