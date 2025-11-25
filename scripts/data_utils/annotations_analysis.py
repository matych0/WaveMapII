import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_time_histogram(path, bins=20, shift=None, shift_gaussian_std=0):

    df = pd.read_csv(path)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.05)

    if shift is not None:
        df.loc[df["reccurence"] == 0, 'days_to_event'] = df.loc[df["reccurence"] == 0, 'days_to_event'] + shift

    sns.histplot(
        ax=ax1,
        data=df, 
        x='days_to_event', 
        hue='reccurence', 
        bins=bins, multiple='layer', 
        palette="vlag", 
        #hue_order=['No', 'Yes'], 
        #edgecolor='black', 
        alpha=0.7,
        )
    
    sns.histplot(
        ax=ax2,
        data=df, 
        x='days_to_event', 
        hue='reccurence', 
        bins=bins, multiple='layer', 
        palette="vlag", 
        #hue_order=['No', 'Yes'], 
        #edgecolor='black', 
        alpha=0.7,
        )
    
    ax2.set_ylabel("Frequency [-]") # , loc='top')
    ax1.set_ylabel("")

    ax2.set_xlabel("Time to event [days]")
    ax1.set_xlabel("")

    ax1.legend(title="", labels=["recurrent", "censored"])
    ax2.legend_.remove()
    
    ax1.set_ylim(165.5, 176)  # outliers only
    ax2.set_ylim(0, 10.5)  # most of the data

    ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)


    plt.show()
    

def plot_time_histogram_with_gaussian(path, bins=20, shift_gaussian_std=0):
    """
    Plot a histogram of the time to event with an optional Gaussian shift.
    """
    df = pd.read_csv(path)
    
    fig, ax = plt.subplots(figsize=(6, 4))

    controls_mask = df['reccurence'] == 0
    
    np_rng = np.random.default_rng(seed=3052001)  # For reproducibility
    
    if shift_gaussian_std > 0:
        noise = np_rng.normal(loc=0, scale=shift_gaussian_std, size=controls_mask.sum())
        noise = np.round(noise).astype(int)
        df.loc[controls_mask, 'days_to_event'] += noise

    sns.histplot(
        ax=ax,
        data=df, 
        x='days_to_event', 
        hue='reccurence', 
        bins=bins, multiple='layer', 
        palette="vlag", 
        #hue_order=['No', 'Yes'], 
        #edgecolor='black', 
        alpha=0.7,
        )
    
    ax.set_ylabel("Frequency [-]") # , loc='top')
    ax.set_xlabel("Time to event [days]")
    ax.legend(title="", labels=["recurrent", "censored"])

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    ANNOTATION_DIR = "D:/HDF5/annotations_complete.csv"

    #plot_time_histogram(ANNOTATION_DIR, bins=25, shift_gaussian_std=20)
    
    plot_time_histogram_with_gaussian(ANNOTATION_DIR, bins=25, shift_gaussian_std=20)
