import numpy as np
from matplotlib import pyplot as plt

from src.transforms.transforms import (RandomAmplifier, RandomGaussian,
                                       RandomShift, RandomTemporalScale,
                                       TanhNormalize)


def create_sinusoidal_tensor(frequency, amplitude, offset, sampling_rate, duration, num_signals):
    """
    Create a torch tensor representing a sinusoidal signal.
    
    :param frequency: Frequency of the sinusoidal signal in Hz
    :param sampling_rate: Sampling rate in Hz
    :param duration: Duration of the signal in seconds
    :return: Torch tensor containing the sinusoidal signal
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    sinusoidal_signal = amplitude[0]*(np.sin(2 * np.pi * frequency[0] * t)) + offset[0]
    for i in range(num_signals - 1):
        sinusoidal_signal = np.vstack((sinusoidal_signal, amplitude[i+1]*(np.sin(2 * np.pi * frequency[i+1] * t)) + offset[i+1]))
                                    
    return sinusoidal_signal


def plot_subplots(data, trans_data, num_subplots, title="Subplots", fs=2035):
    """
    Plot subplots using Matplotlib.
    
    :param data: List of data arrays to plot
    :param num_subplots: Number of subplots
    :param title: Title of the plot
    """
    fig, ax = plt.subplots(num_subplots, 2, figsize=(10, 10))
    ax = ax.flatten() 
    fig.suptitle(title)

    data_min = min(data.min(), trans_data.min())
    data_max = max(data.max(), trans_data.max())
    padding = 0.2

    time_ms = np.arange(data.shape[1]) * (1000 / fs)
    
    for i in range(2*num_subplots):
        if i % 2 == 0:
            ax[i].plot(time_ms, data[i//2])
            ax[i].set_ylim(data_min - padding, data_max + padding)
            ax[i].axhline(0, color='red', linestyle='--', linewidth=1)
            #ax[i].set_title(f"Signal {i//2+1}")
            ax[i].set_xlabel("time [ms]")
            ax[i].set_ylabel("voltage [mV]")
        else:
            ax[i].plot(time_ms, trans_data[i//2])
            ax[i].set_ylim(-1 - padding, 1 + padding)
            ax[i].axhline(0, color='red', linestyle='--', linewidth=1)
            #ax[i].set_title(f"Transformed Signal {i//2+1}")
            ax[i].set_xlabel("time [ms]")
            ax[i].set_ylabel("amplitude [mV]")

    

    plt.tight_layout()
    plt.show()

def visualize_transforms(signal, seed):
    # Define two arbitrary settings for each transform
    arbitrary_values = {
        "temporal_scale": [0.8, 1.2],
        "amplifier": [0.8, 1.2],
        "noise": [10, 30],
        "shift": [-0.3, 0.3]
    }

    # Create transform instances with different arbitrary values
    transforms = {
        "Temporal Scaling": [
            RandomTemporalScale(probability=1, limit=0.2, arbitrary=arbitrary_values["temporal_scale"][0], shuffle=True, random_seed=seed),
            RandomTemporalScale(probability=1, limit=0.2, arbitrary=arbitrary_values["temporal_scale"][1], shuffle=True, random_seed=seed)
        ],
        "Amplitude Scaling": [
            RandomAmplifier(probability=1, limit=0.2, arbitrary=arbitrary_values["amplifier"][0], shuffle=True, random_seed=seed),
            RandomAmplifier(probability=1, limit=0.2, arbitrary=arbitrary_values["amplifier"][1], shuffle=True, random_seed=seed)
        ],
        "Jittering": [
            RandomGaussian(probability=1, low_limit=10, high_limit=30, arbitrary=arbitrary_values["noise"][0], shuffle=True, random_seed=seed),
            RandomGaussian(probability=1, low_limit=10, high_limit=30, arbitrary=arbitrary_values["noise"][1], shuffle=True, random_seed=seed)
        ],
        "Temporal Shifting": [
            RandomShift(probability=1, shift_range=0.3, arbitrary=arbitrary_values["shift"][0], shuffle=True, random_seed=seed),
            RandomShift(probability=1, shift_range=0.3, arbitrary=arbitrary_values["shift"][1], shuffle=True, random_seed=seed)
        ]
    }

    # Plot setup
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(7, 11))

    data_min = signal.min() * arbitrary_values["amplifier"][1]
    data_max = signal.max() * arbitrary_values["amplifier"][1]
    padding = 0.2

    for i, (name, (transform_low, transform_high)) in enumerate(transforms.items()):
        low = transform_low.arbitrary
        high = transform_high.arbitrary

        if name in ["Temporal Scaling", "Amplitude Scaling", "Temporal Shifting"]:
            title_low = f"{name}: {int(low*100)} %"
            title_high = f"{name}: {int(high*100)} %"
        else:
            title_low = f"{name}: {low} dB"
            title_high = f"{name}: {high} dB"

        if name == ("Temporal Shifting"):
            title_high = f"{name}: +{int(high*100)} %"

        time_ms = np.arange(signal.shape[1]) * (1000 / 2035)

        # Plot original
        axes[i, 0].plot(time_ms, signal[0,:], color='black')
        axes[i, 0].set_title("Original Signal", fontsize=11)
        axes[i, 0].set_ylabel("Voltage [mV]")
        axes[i, 0].set_ylim(data_min - padding, data_max + padding)

        # Plot with lower arbitrary value
        transformed1 = transform_low(signal.copy())
        axes[i, 1].plot(time_ms, transformed1[0,:], color='blue')
        axes[i, 1].set_title(title_low, fontsize=11)
        axes[i, 1].set_ylim(data_min - padding, data_max + padding)

        # Plot with higher arbitrary value
        transformed2 = transform_high(signal.copy())
        axes[i, 2].plot(time_ms, transformed2[0,:], color='red')
        axes[i, 2].set_title(title_high, fontsize=11)
        axes[i, 2].set_ylim(data_min - padding, data_max + padding)

        if i == 3:
            axes[i, 0].set_xlabel("Time [ms]")
            axes[i, 1].set_xlabel("Time [ms]")
            axes[i, 2].set_xlabel("Time [ms]")

        for j in range(3):
            axes[i, j].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    


def visualize_single_transform_two_signals(signal1, signal2, transform, sampling_rate=2035):
    """
    Visualizes the effect of a single transform on two different signals in a 2x2 plot.
    
    Parameters:
    - signal1, signal2: Input signals of shape (1, T)
    - transform: A transform instance with __call__ method, e.g., TanhNormalize(factor=5)
    - sampling_rate: Sampling rate in Hz to convert time axis to milliseconds
    """

    signals = [signal1, signal2]
    time_ms_1 = np.arange(signal1.shape[1]) * (1000 / sampling_rate)
    time_ms_2 = np.arange(signal2.shape[1]) * (1000 / sampling_rate)
    time_axes = [time_ms_1, time_ms_2]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))

    for i, (sig, time_ms) in enumerate(zip(signals, time_axes)):
        # Original signal
        orig_signal = sig[0]
        sig_min = orig_signal.min()
        sig_max = orig_signal.max()
        original_padding = np.abs(sig_max - sig_min) * 0.2
        axes[i, 0].plot(time_ms, orig_signal, color='black')
        if i == 0:
            axes[i, 0].set_title("Original Signal", fontsize=11)
        axes[i, 0].set_ylabel("Voltage [mV]")
        axes[i, 0].set_ylim(sig_min - original_padding, sig_max + original_padding)
        axes[i, 0].grid(True)

        # Transformed signal
        transformed = transform(sig.copy())
        trans_signal = transformed[0]
        trans_min = trans_signal.min()
        trans_max = trans_signal.max()
        transformed_padding = np.abs(trans_max - trans_min) * 0.2

        axes[i, 1].plot(time_ms, trans_signal, color='blue')
        if i == 0:
            axes[i, 1].set_title(f"Tanh Normalization", fontsize=11)
        axes[i, 1].set_ylim(trans_min - transformed_padding, trans_max + transformed_padding)
        axes[i, 1].grid(True)

    for j in range(2):
        axes[1, j].set_xlabel("Time [ms]")

    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    """ frequencies = [1, 2, 3, 4, 17]
    amplitudes = [1.2, 0.8, 1.5, 1.0, 0.7]
    offsets = [0.5, 0.3, 0.7, 0.2, -0.5]
    x = create_sinusoidal_tensor(frequency=frequencies,amplitude=amplitudes,offset=offsets, sampling_rate=200, duration=1, num_signals=len(frequencies))
 """


    ANNOTATION_DIR = "/media/guest/DataStorage/WaveMap/HDF5/annotations_train.csv"
    DATA_DIR = "/media/guest/DataStorage/WaveMap/HDF5"
    
    from torch.utils.data import DataLoader

    from src.dataset.dataset import HDFDataset
    
    training_data = HDFDataset(
        annotations_file=ANNOTATION_DIR,
        data_dir=DATA_DIR,
        train=True,
        transform=None,            
        startswith="LA",
        readjustonce=True, 
        num_traces=100,
        segment_ms=100,
        filter_utilized=True
    )


    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=False)

    x, y = next(iter(train_dataloader))
    
    

    x_orig = np.expand_dims(np.array(x[0,95,:]), axis=0)

    y_orig = np.expand_dims(np.array(y[0,74,:]), axis=0)

    seed = 5032001

    """ #polarity =  RandomPolarity(probability=0.5, shuffle=True, random_seed=seed)
    temporal_scale = RandomTemporalScale(probability=0.2, limit=0.2, shuffle=True, random_seed=seed)
    amplifier = RandomAmplifier(probability=0.4, limit=0.5, shuffle=True, random_seed=seed)
    noise = RandomGaussian(probability=0.2, low_limit=10, high_limit=30, shuffle=True, random_seed=seed)
    shift = RandomShift(probability=0.8, shuffle=True, random_seed=seed, shift_range=0.3)
    tanh_normalize = TanhNormalize(factor=5)
    shuffle = BaseTransform(shuffle=True, random_seed=seed)

    transform = transforms.Compose([
        #polarity,
        amplifier,
        temporal_scale,
        shift,
        noise,
        shuffle,
        tanh_normalize,
    ])

    x_trans = transform(x)

    plot_subplots(x, x_trans, num_subplots=x_orig.shape[0], title="Signal Transform") """

    """ x = np.array(x_orig[1,95:,:])
    x_trans_2 = transform(x)

    plot_subplots(x_orig, x_trans_2, num_subplots=x_orig.shape[0], title="Signal Transform")
    
 """
    
    #visualize_transforms(x, seed)

    tanh_normalize = TanhNormalize(factor=5)

    visualize_single_transform_two_signals(x_orig, y_orig, tanh_normalize, sampling_rate=2035)

    print("Done")