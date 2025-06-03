import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_time_histogram(path, bins=20):

    df = pd.read_csv(path)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.05)

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
    
    ax1.set_ylim(101.5, 112)  # outliers only
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


def plot_with_broken_axis(path, bins=20):

    df = pd.read_csv(path)

    # Create synthetic data with one high-frequency value
    np.random.seed(0)
    times = np.concatenate([np.random.randint(0, 40, 100), np.repeat(10, 100)])
    events = np.random.choice([0, 1], size=len(times))

    df = pd.DataFrame({'time': times, 'event': events})

    # Create two vertically stacked subplots (shared x-axis)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), 
                                gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05})

    # Plot the same histogram on both axes
    sns.histplot(data=df, x='time', hue='event', bins=30, multiple='stack', ax=ax1)
    sns.histplot(data=df, x='time', hue='event', bins=30, multiple='stack', ax=ax2)

    # Set different y-limits
    ax1.set_ylim(80, 120)  # only show the top part of tall bars
    ax2.set_ylim(0, 30)    # show the smaller bars

    # Hide spines between axes
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False)  # don't show x-ticks on top plot

    # Add diagonal break marks
    d = .005  # size of diagonal lines
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # Label axes
    ax2.set_xlabel("Time")
    ax1.set_ylabel("Count")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    ANNOTATION_DIR = "/media/guest/DataStorage/WaveMap/HDF5/annotations_train.csv"

    plot_time_histogram(ANNOTATION_DIR, bins=20)

    #plot_with_broken_axis(ANNOTATION_DIR, bins=20)