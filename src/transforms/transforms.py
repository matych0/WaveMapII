import random
import itertools
from abc import ABC, abstractmethod
import numpy as np
from scipy import signal
from scipy.signal import firwin, filtfilt, butter
import json
from matplotlib import pyplot as plt
from scipy.signal import resample
from scipy.stats import loguniform
import torch


__all__ = ["Compose", "HardClip", "ZScore", "RandomShift", "RandomStretch", "RandomAmplifier", "RandomLeadSwitch",
           "Resample", "BaseLineFilter", "OneHotEncoding", "AddEmgNoise"
           ]


# ---------------------------- Composing class -----------------------------
# --------------------------------------------------------------------------

def read_header(file_name):
    """Returns dict containing x information"""
    with open(file_name, "r") as file:
        return json.load(file)


class Compose(object):
    """Composes several transforms together.
    Example:
        transforms.Compose([
            transforms.HardClip(10),
            transforms.ToTensor(),
            ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y, **kwargs):
        if self.transforms:
            for t in self.transforms:
                x, y = t(x, **kwargs)

        return x, y

    def reset_state(self):
        """
        Method invoke local random generator resetter.
        :return:
        """
        for t in self.transforms:
            if hasattr(t, 'reset_state'):
                t.reset_state()


# -------------------- Classes for data transformations --------------------
# --------------------------------------------------------------------------

class BaseTransform:
    def __init__(self, shuffle=False, random_seed=None, **kwargs):
        self.shuffle = shuffle  
        self.np_rng = np.random.default_rng(random_seed) # NumPy random generator

    def __call__(self, x, **kwargs):
        if self.shuffle:
            self.np_rng.shuffle(x)     
        return self.transform(x, **kwargs)

    def transform(self, x, **kwargs):
        return x  # Default: return input unchanged


class ZScore(BaseTransform):
    """Returns Z-score normalized data"""
    def __init__(self, mean: float = 0, std: float = 1000, **kwargs):
        super().__init__()

        self.mean = mean
        self.std = std

    def __call__(self, x, **kwargs):
        return super().__call__(x)

    def transform(self, x, **kwargs):
        x = (x - self.mean) / self.std

        return x
    

class Normalize(BaseTransform):
    """Normalizes each signal to range [-1, 1]"""
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, x, **kwargs):
        return super().__call__(x)

    def transform(self, x, **kwargs):
        x_min = np.min(x, axis=1, keepdims=True)
        x_max = np.max(x, axis=1, keepdims=True)
        x = (x - x_min ) / (x_max - x_min + 1e-8) * 2 - 1

        return x


# --------------------- Classes for data augmentation ----------------------
# --------------------------------------------------------------------------


class RandomPolarity(BaseTransform):
    """
        Class randomly switches signal polarity
    """
    def __init__(self, probability: float, shuffle: bool = True, random_seed: int = None, **kwargs):
        super().__init__(shuffle=shuffle, random_seed=random_seed)
        self.probability = probability

    def __call__(self, x, **kwargs):
        return super().__call__(x)

    def transform(self, x, **kwargs):
        num_signals = x.shape[0]
        num_transformed = int(self.probability * num_signals)  # Determine how many signals to zero
        x[:num_transformed, :] *= -1.0  # Apply polarity switch only to the first 'num_transformed' signals
        return x


class RandomShift(BaseTransform):
    """
        Class randomly shifts signal within temporal dimension
    """
    def __init__(self, probability: float, shift_range: float = 0.9, shuffle: bool = True, random_seed: int = None, **kwargs):
        assert 0.0 < shift_range <= 1.0, "shift_range should be in (0, 1]."
        
        self.probability = probability
        self.shift_range = shift_range
        self.random_seed = random_seed
        super().__init__(shuffle=shuffle, random_seed=random_seed)


    def __call__(self, x, **kwargs):
        return super().__call__(x)

    def transform(self, x, **kwargs):
        np.random.seed(self.random_seed)  # Ensure reproducibility if seed is set

        num_signals, signal_length = x.shape
        num_transformed = round(self.probability * num_signals)  # Number of signals to shift

        # Generate random shift fractions in range [-shift_range, shift_range]
        shift_fractions = np.random.uniform(-self.shift_range, self.shift_range, size=num_transformed)

        # Convert to integer shift values
        shifts = np.round(shift_fractions * signal_length).astype(int)

        # Apply shifts using np.roll
        for i, shift in zip(range(num_transformed), shifts):
            x[i] = np.roll(x[i], shift, axis=-1)  # Shift along time axis

        return x
    

class RandomGaussian(BaseTransform):
    """
        Class randomly adds gaussian noise to signal
    """
    def __init__(self, probability: float, low_limit: float, high_limit: float, shuffle: bool = True, random_seed: int = None, **kwargs):
        super().__init__(shuffle=shuffle, random_seed=random_seed)
        self.probability = probability
        self.random_seed = random_seed
        self.low_limit = low_limit
        self.high_limit = high_limit
        

    def __call__(self, x, **kwargs):
        return super().__call__(x)

    def transform(self, x, **kwargs):
        np.random.seed(self.random_seed)  # Ensure reproducibility if seed is set

        num_signals, signal_length = x.shape
        num_transformed = round(self.probability * num_signals)  # Number of signals to add noise to

        noise_dbs = np.random.uniform(self.low_limit, self.high_limit, size=num_transformed)

        for i, db in zip(range(num_transformed), noise_dbs):
            # estimate SNR
            power = np.sqrt(np.mean(x[i, :] ** 2) / (10 ** (db / 10)))
            # add noise
            x[i] += np.random.normal(loc=0, scale=power, size=signal_length)
        return x


class RandomAmplifier(BaseTransform):
    """
    Class randomly amplifies signal
    """
    def __init__(self, probability: float, limit: float, shuffle: bool = True, random_seed: int = None, **kwargs):
        super().__init__(shuffle=shuffle, random_seed=random_seed)

        self.random_seed = random_seed
        self.probability = probability
        self.limit = limit

    def __call__(self, x, **kwargs):
        return super().__call__(x, **kwargs)


    def transform(self, x, **kwargs):
        np.random.seed(self.random_seed)  # Ensure reproducibility if seed is set

        num_signals, signal_length = x.shape
        num_transformed = round(self.probability * num_signals)  # Number of signals to amplify

        amplify_factors = np.random.uniform(-self.limit, self.limit, size=num_transformed)
        print(amplify_factors)
        x[:num_transformed] *= amplify_factors[:, np.newaxis]

        return x

class RandomStretch(BaseTransform):
    """
    Class randomly stretches temporal dimension of signal
    """
    def __init__(self, probability: float, limit: float, shuffle: bool = True, random_seed: int = None, **kwargs):
        super().__init__(shuffle=shuffle, random_seed=random_seed)

        self.random_seed = random_seed
        self.probability = probability
        self.limit = limit

    def __call__(self, x, **kwargs):
        return super().__call__(x, **kwargs)

    def transform(self, x, **kwargs):
        np.random.seed(self.random_seed)  # Ensure reproducibility if seed is set
        num_signals, width = x.shape
        num_transformed = round(self.probability * num_signals)  # Number of signals to amplify
        
        factors = np.random.uniform(1/self.limit, self.limit, size=num_transformed)
        new_widths = (factors * width).astype(int)
        print(new_widths)

        for i, new_w in zip(range(num_transformed), new_widths):
            if new_w > width:
                xp_new = np.linspace(0, width - 1, width)
                xp = np.linspace(0, new_w - 1, width)                
            elif new_w < width:
                xp_new = np.linspace(0, new_w - 1, width)
                xp = np.linspace(0, width - 1, width)
            else:
                continue
            
            x[i, :] = np.interp(
                xp_new,
                xp,
                x[i, :],
            )
        return x


class RandomTemporalScale(BaseTransform):
    """
    Class randomly stretches temporal dimension of signal
    """
    def __init__(self, probability: float, limit: float, shuffle: bool = True, random_seed: int = None, **kwargs):
        super().__init__(shuffle=shuffle, random_seed=random_seed)

        self.random_seed = random_seed
        self.probability = probability
        self.limit = limit

    def __call__(self, x, **kwargs):
        return super().__call__(x, **kwargs)

    def transform(self, x, **kwargs):
        np.random.seed(self.random_seed)  # Ensure reproducibility if seed is set
        num_signals, width = x.shape
        num_transformed = round(self.probability * num_signals)  # Number of signals to amplify

        factors = np.random.uniform(1/self.limit, self.limit, size=num_transformed)
        new_widths = (factors * width).astype(int)
        print(new_widths)

        x_resampled = np.zeros_like(x)

        for i, new_w in zip(range(num_transformed), new_widths):
            resampled_signal = resample(x[i], new_w)  # Resample

            # Compute padding or cropping offsets
            diff = width - new_w
            if diff < 0:  # Crop equally from both sides
                start = -diff // 2
                end = start + width
                x_resampled[i] = resampled_signal[start:end]
            else:  # Pad equally on both sides
                pad_left = diff // 2
                pad_right = diff - pad_left
                x_resampled[i, pad_left:-pad_right or None] = resampled_signal

        # Copy the remaining signals unchanged
        x_resampled[num_transformed:] = x_resampled[num_transformed:]

        return x_resampled


#----------------------Others------------------------------------------------
#----------------------------------------------------------------------------
    
class HardClip(BaseTransform):
    """Returns scaled and clipped data between range <-clipping_threshold:clipping_threshold>"""
    def __init__(self, threshold: float, use_on='sample', **kwargs):
        super().__init__(use_on)
        self.threshold = threshold
        self.generator = False

    def __call__(self, x, y, **kwargs):
        return super().__call__(x, y)

    def transform_x(self, x, **kwargs):
        x[x > self.threshold] = self.threshold
        x[x < -self.threshold] = -self.threshold

        return x

    
class HighPassFilter(BaseTransform):
    def __init__(self, critical_f: float, sampling_f: int, order: int, use_on: str, **kwargs):
        super().__init__(use_on)

        self.filter = signal.firwin(
            order,
            cutoff=critical_f,
            window="nuttall",
            pass_zero=False,
            fs=sampling_f,
        )

    def __call__(self, x, y, **kwargs):
        return super().__call__(x, y, **kwargs)

    def transform_x(self, x, **kwargs):
        y_t = np.zeros(x.shape)
        for i, row in enumerate(x):
            y_t[i, :] = signal.fftconvolve(row, self.filter, mode="same")
        return y_t


class LowPassFilter(BaseTransform):
    def __init__(self, critical_f: float, sampling_f: int, order: int = 6, use_on='sample', **kwargs):
        super().__init__(use_on)

        self.filter = signal.firwin(
            order,
            cutoff=critical_f,
            window="nuttall",
            pass_zero=True,
            fs=sampling_f,
        )

    def __call__(self, x, y, **kwargs):
        return super().__call__(x, y, **kwargs)

    def transform_x(self, x, **kwargs):
        y_t = np.zeros(x.shape)
        for i, row in enumerate(x):
            y_t[i, :] = signal.fftconvolve(row, self.filter, mode="same")
        return y_t
    
    
class RandomZeroing(BaseTransform):
    """ Class randomly zeros signals """

    def __init__(self, probability: float, shuffle: bool = True, random_seed: int = None, **kwargs):
        super().__init__(shuffle=shuffle, random_seed=random_seed)

        self.probability = probability

    def __call__(self, x, **kwargs):
        return super().__call__(x)

    def transform(self, x, **kwargs):
        num_signals = x.shape[0]
        num_transformed = int(self.probability * num_signals)  # Determine how many signals to zero
        x[:num_transformed, :] = 0.0  # Apply zeroing only to the first 'num_zeroed' signals
        return x
    

class RandomArtifact(BaseTransform):
    """
        Class randomly adds stimulation artifact
    """
    def __init__(self, probability: float, use_on: str, **kwargs):
        super().__init__(use_on)
        self.probability = probability

    def __call__(self, x, y, **kwargs):
        return super().__call__(x, y)

    def transform_x(self, x, **kwargs):
        dice = self.rng.uniform(0, 1)
        if dice < self.probability:
            h, w = x.shape
            # cycle_length 200-2000 ms
            cl = self.rng.randint(400, 4000)

            # peak plato 2-10 ms
            plato_t = self.rng.randint(4, 20)
            # decay phase 5-30 ms
            decay_t, decay_e = self.rng.randint(10, 60), self.rng.randint(-10, -2)
            # generate shape
            shape = np.concatenate(
                (
                    np.array([0, 0, 0.17, 1]),
                    np.ones(plato_t),
                    np.exp(np.linspace(0, decay_e, decay_t)),
                    np.zeros(2),
                ),
                axis=0,
            )
            shape = np.tile(shape, reps=(h, 1))
            # generate amplitudes
            multipliers = np.random.uniform(low=-500, high=5000, size=(h, 1))
            shape = shape * multipliers
            shape_w = shape.shape[-1]
            for i in range(shape_w, w - shape_w, cl):
                x[:, i:i+shape_w] += shape

        return x


#----------------------Plotting--------------------------------------------
#--------------------------------------------------------------------------


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


def plot_subplots(data, trans_data, num_subplots, title="Subplots"):
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
    
    for i in range(2*num_subplots):
        if i % 2 == 0:
            ax[i].plot(data[i//2])
            ax[i].set_ylim(data_min - padding, data_max + padding)
            ax[i].axhline(0, color='red', linestyle='--', linewidth=1)
            ax[i].set_title(f"Signal {i//2+1}")
            ax[i].set_xlabel("Samples")
            ax[i].set_ylabel("Amplitude")
        else:
            ax[i].plot(trans_data[i//2])
            ax[i].set_ylim(data_min - padding, data_max + padding)
            ax[i].axhline(0, color='red', linestyle='--', linewidth=1)
            ax[i].set_title(f"Transformed Signal {i//2+1}")
            ax[i].set_xlabel("Samples")
            ax[i].set_ylabel("Amplitude")

    

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    frequencies = [1, 2, 3, 4, 17]
    amplitudes = [1.2, 0.8, 1.5, 1.0, 0.7]
    offsets = [0.5, 0.3, 0.7, 0.2, -0.5]
    x = create_sinusoidal_tensor(frequency=frequencies,amplitude=amplitudes,offset=offsets, sampling_rate=200, duration=1, num_signals=len(frequencies))



    """ data = annotation_filepath = "/home/guest/lib/data/WaveMapSampleHDF/event_data.csv"
    dataset_folderpath = "/home/guest/lib/data/WaveMapSampleHDF"
    
    from torch.utils.data import DataLoader
    from src.dataset.dataset import HDFDataset
    
    training_data = HDFDataset(
        annotations_file=annotation_filepath,
        data_dir=dataset_folderpath,
        train=True,
        transform=None,            
        startswith="LA",
        readjustonce=False, 
        num_traces=5,
        segment_ms=100
    )

    train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True)

    x, y = next(iter(train_dataloader)) """

    x_orig = np.copy(x)
    #transform = RandomZeroing(probability=0.19, shuffle=True, random_seed=42)
    #transform = RandomPolarity(probability=0.5, shuffle=True, random_seed=42)
    #transform = RandomShift(probability=0.5, shuffle=True, random_seed=42)
    #transform = RandomGaussian(probability=0.5, low_limit=10, high_limit=40, shuffle=True, random_seed=42)
    #transform = RandomArtifact(probability=1, use_on="sample")
    #transform = RandomPowerline(probability=0.7, use_on="sample", low_limit=5, high_limit=10)
    # transform = RandomCrop(probability=0.7, use_on="sample", limit=20)
    # transform = RandomAmplifier(probability=1, limit=2, shuffle=True, random_seed=42)
    # transform = RandomStretch(probability=1, limit=2, shuffle=True, random_seed=42)
    transform = RandomTemporalScale(probability=1, limit=2, shuffle=True, random_seed=42)
    #transform = ZScore(mean=1, std = 2)
    # transform = Normalize()
    x_trans = transform(x)
    #x_trans = transform(x_trans)
    
    
    plot_subplots(x_orig, x_trans, num_subplots=x_orig.shape[0], title="Signal Transform")
    

