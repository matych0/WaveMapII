#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind, kstest

#%%
csv_path = "C:/Users/marti/Documents/Diplomka/results/post_analysis/inference_df_export.csv"

df = pd.read_csv(csv_path)

#%%
""" def label_attention(x):
    #if x >= 0.0021085963
        #return 'Very High'
    if x > 0.0015496228:
        return '99th percentile and above'
    elif x > 0.0011487881:
        return '95-99th percentile'
    else:
        return '0-95th percentile'
    
df['Attention Level'] = df['attention'].apply(label_attention) """

#%%
df['Attention Level'] = pd.qcut(df['attention'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
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
g.add_legend(title="Attention Level", bbox_to_anchor=(0.5, 0.95), loc='center', ncol=4)

#plt.savefig('boxplot_all_parameters_three_cat.svg', format='svg')
plt.tight_layout()
plt.show()

#%%
# Statistical tests

results = []

reference_level = "0-95th percentile"

for param in df_melted['parameter'].unique():
    df_param = df_melted[df_melted['parameter'] == param]
    ref_values = df_param[df_param['Attention Level'] == reference_level]['value']

    for level in df_param['Attention Level'].unique():
        if level == reference_level:
            continue

        group_values = df_param[df_param['Attention Level'] == level]['value']

        # Standardize values for K-S test (mean 0, std 1)
        try:
            ref_std = (ref_values - ref_values.mean()) / ref_values.std()
            grp_std = (group_values - group_values.mean()) / group_values.std()

            ref_normal = kstest(ref_std, 'norm').pvalue > 0.05
            grp_normal = kstest(grp_std, 'norm').pvalue > 0.05
        except:
            ref_normal = grp_normal = False  # fallback for very small n or numerical issues

        if ref_normal and grp_normal:
            # Use Welch's t-test
            stat, p = ttest_ind(ref_values, group_values, equal_var=False)
            test_type = "Welch t-test"
        else:
            # Use Mann-Whitney U test
            stat, p = mannwhitneyu(ref_values, group_values, alternative='two-sided')
            test_type = "Mann-Whitney U"

        results.append({
            'parameter': param,
            'comparison': f'{reference_level} vs {level}',
            'p-value': p,
            'test': test_type
        })

results_df = pd.DataFrame(results)
print(results_df)

#%%
""" plt.figure(figsize=(4,7))
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
plt.show() """

