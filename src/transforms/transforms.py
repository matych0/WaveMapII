import random
import itertools
from abc import ABC, abstractmethod
import numpy as np
from scipy import signal
from scipy.signal import firwin, filtfilt, butter
import json
from matplotlib import pyplot as plt
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
                x, y = t(x, y, **kwargs)

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
    def __init__(self, mean: float = 0, std: float = 1000, use_on='sample', **kwargs):
        super().__init__(use_on)

        self.mean = mean
        self.std = std

    def __call__(self, x, y, **kwargs):
        return super().__call__(x, y)

    def transform_x(self, x, **kwargs):
        x = x - np.array(self.mean).reshape(-1, 1)
        x = x / self.std

        return x


class ResampleSignals(BaseTransform):
    def __init__(
            self,
            input_sampling: int,
            output_sampling: int,
            use_on: str,
            **kwargs
    ):
        super().__init__(use_on)
        self.input_sampling = int(input_sampling)
        self.output_sampling = int(output_sampling)

    def __call__(self, x, y, **kwargs):
        return super().__call__(x, y, **kwargs)

    def transform_x(self, x, **kwargs):
        return signal.resample_poly(
            x,
            up=1,
            down=self.input_sampling // self.output_sampling,
            axis=1,
            window='nuttall',
            padtype='constant',
            cval=None,
        )


class ResampleTargets(BaseTransform):
    def __init__(
            self,
            input_sampling: int,
            output_sampling: int,
            use_on: str,
            **kwargs
    ):
        super().__init__(use_on)
        self.input_sampling = int(input_sampling)
        self.output_sampling = int(output_sampling)

    def __call__(self, x, y, **kwargs):
        if self.input_sampling == self.output_sampling:
            return x, y
        return super().__call__(x, y, **kwargs)

    def transform_y(self, y, **kwargs):
        if y is None:
            return y

        factor = self.output_sampling / self.input_sampling
        y_t = []
        for mark, intervals in y:
            intervals = [[int(l * factor), int(r * factor)] for l, r in intervals]
            y_t.append((mark, intervals))

        return y_t


class ExtendTargets(BaseTransform):
    def __init__(self, width: int, use_on: str, **kwargs):
        super().__init__(use_on)
        self.width = int(width)

    def __call__(self, x, y, **kwargs):
        return super().__call__(x, y, **kwargs)

    def transform_y(self, y, **kwargs):
        y_t = []
        for mark, intervals in y:
            temp = []
            for l, r in intervals:
                complement = self.width - (r - l)
                l -= complement if complement > 0 else 0
                l = 0 if l < 0 else l
                temp.append([l, r])
            y_t.append((mark, temp))

        return y_t


class PrepareTargets:
    def __init__(
        self,
        cropped_size: int = 1250,
        undersampling_factor: int = 4,
        window_size: int = 17,
        stride: int = 8,
        limit: float = 0.5,
        tolerance: int = 50,
        **kwargs,
    ):

        self.cropped_size = cropped_size
        self.factor = undersampling_factor
        self.window_size = window_size
        self.stride = stride
        self.limit = limit
        self.tolerance = tolerance

        # Right index of each ROI
        self.ROI_max = list(range(self.window_size, self.cropped_size + self.window_size, self.stride))
        # Left index of each ROI
        self.ROI_min = [item - self.window_size for item in self.ROI_max]

    def transform(self, y, **kwargs):
        batched_targets = []        
        downsample = kwargs.pop('downsample', False)

        # reduce and subsample positions
        for target in y:
            t1 = []
            for l_mark, r_mark in target:
                if downsample:
                    r_mark = (1 / self.factor) * r_mark
                    l_mark = (1 / self.factor) * l_mark
                center = (r_mark + l_mark) / 2
                t1.append([l_mark, r_mark, center])
                
            batched_targets.append(t1)

        return batched_targets 

    def to_relative(self, y, **kwargs):                      

        ROI_num = len(self.ROI_max)

        one_hot_classes = np.zeros((len(y), 1, ROI_num))
        prob_classes = np.zeros((len(y), 1, ROI_num))
        norm_positions = 0.0 * np.ones((len(y), 1, ROI_num))
        norm_box_intervals = np.zeros((len(y), 2, ROI_num))
                
        for j, (roi_left, roi_right) in enumerate(zip(self.ROI_min, self.ROI_max)):            
            for i, target in enumerate(y):
                for mark_left, mark_right, center in target:
                    # expand to match P wave duration
                    # !!!!! only for normal P wave                    
                    expansion = (self.tolerance - (mark_right - mark_left))
                    mark_left -= expansion - 10
                    mark_right += 10
                    
                    if mark_left >= roi_left and mark_left < roi_right:
                        one_hot_classes[i, 0, j] = 1.0
                        norm_positions[i, 0, j] = (center - roi_left ) / (roi_right - roi_left)

                    if mark_right >= roi_left and mark_right < roi_right:
                        one_hot_classes[i, 0, j] = 1.0
                        norm_positions[i, 0, j] = (center - roi_left ) / (roi_right - roi_left)

                    # if center >= roi_left and center < roi_right:
                    #     intersection = max(0, min(roi_right, center+7) - max(roi_left, center-7))
                    #     pok[i, 0, j] = intersection / (roi_right - roi_left)                        

                    if mark_left > roi_right:
                        break

        return one_hot_classes, prob_classes, norm_positions

    def to_absolute(self, box_positions, mask, **kwargs):        
        # one_hot_classes = one_hot_classes >= self.limit
        #TODO: max-poolnout po sobe jdouci hodnoty, nechat jen ty po soobe jdouci s nejvyssi pravdepodonosti
        # vyskytu P vlny.
        upsample = kwargs.pop('upsample', False)

        if mask is not None:
            box_positions = box_positions * mask
        
        batch_n, class_n, _ = box_positions.shape
        roi_min = np.tile(np.array(self.ROI_min), (batch_n, class_n, 1))
        roi_max = np.tile(np.array(self.ROI_max), (batch_n, class_n, 1))

        box_positions = roi_min + ((box_positions * roi_max) - roi_min * (1 - box_positions))
        
        if upsample:
            box_positions = box_positions * self.factor

        return box_positions


class OneHotEncoding:
    """Returns one hot encoded labels"""
    def __init__(self, cropped_size: int, factor: float):
        self.cropped_size = cropped_size
        self.factor = factor
        self.duration = 25

    def __call__(self, targets, **kwargs):
        expand_targets = kwargs.pop('expand_targets', False)
        cropped_size = kwargs.pop('cropped_size', None)

        if self.cropped_size is None:
            y_t = np.zeros([len(targets), cropped_size])
        else:
            y_t = np.zeros([len(targets), self.cropped_size])

        for i, intervals in enumerate(targets):
            for lower_mark, upper_mark in intervals:
                lower_mark = lower_mark // self.factor
                upper_mark = -(upper_mark // -self.factor)

                if expand_targets:
                    diff = upper_mark - lower_mark
                    if diff < self.duration:
                        lower_mark = lower_mark - (self.duration - diff)
                        if lower_mark < 0:
                            lower_mark = 0

                y_t[i, lower_mark:upper_mark] = 1.0

        return y_t


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


# --------------------- Classes for data augmentation ----------------------
# --------------------------------------------------------------------------

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


class RandomPowerline(BaseTransform):
    """
        Class randomly shifts signal within temporal dimension
    """
    def __init__(self, probability: float, low_limit: float, high_limit: float, use_on: str, **kwargs):
        super().__init__(use_on)
        self.probability = probability
        self.low_limit = low_limit
        self.high_limit = high_limit

    def __call__(self, x, y, **kwargs):
        return super().__call__(x, y)

    def transform_x(self, x, **kwargs):
        phase = self.rng.uniform(0, 1)
        for i in range(x.shape[0]):
            dice = self.rng.uniform(0, 1)
            if dice < self.probability:
                db = self.rng.uniform(self.low_limit, self.high_limit)
                # estimate SNR
                power = np.sqrt(np.mean(x[i, :] ** 2) / (10 ** (db / 10)))
                # add noise
                sine = (1.414 * power) * np.sin(2 * np.pi * 50 * np.arange(0, (x.shape[-1] + 1) / 2000, 1 / 2000) + phase)
                x[i, :] += sine[:x.shape[-1]]
        return x


class RandomCrop(BaseTransform):
    """
        Class randomly shifts signal within temporal dimension
    """
    def __init__(self, probability: float, limit: float, use_on: str, **kwargs):
        super().__init__(use_on)
        assert limit > 0

        self.probability = probability
        self.limit = limit

    def __call__(self, x, y, **kwargs):
        dice = self.rng.uniform(0, 1)
        if dice < self.probability:
            w = x.shape[-1]
            lower_bound = self.rng.randint(0, self.limit)
            upper_bound = w - self.rng.randint(0, self.limit)
            return super().__call__(x, y, lower_bound=lower_bound, upper_bound=upper_bound)
        else:
            return x, y

    def transform_x(self, x, **kwargs):
        lower_bound = kwargs.pop('lower_bound', 0)
        upper_bound = kwargs.pop('upper_bound', x.shape[-1])
        return x[:, lower_bound:upper_bound]

    def transform_y(self, y,  **kwargs):
        if y is None:
            return y

        lower_bound = kwargs.pop('lower_bound')
        upper_bound = kwargs.pop('upper_bound')

        y_t = list()
        for mark_tag, mark_vals in y:
            temp = list()
            for lower_mark, upper_mark in mark_vals:
                lower_mark -= lower_bound
                upper_mark -= lower_bound

                # skip cropped marks
                if upper_mark < 0:
                    continue

                if lower_mark >= upper_bound:
                    continue

                # check if interval has dropped below lower bound
                if lower_mark < 0:
                    lower_mark = 0

                # check if interval is above upper bound
                if upper_mark >= upper_bound:
                    upper_mark = upper_bound - 1

                temp.append([lower_mark, upper_mark])

            y_t.append(
                (mark_tag, sorted(temp))
            )

        return y_t


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
    def __init__(self, probability: float, limit: float, use_on='sample', **kwargs):
        super().__init__(use_on)

        self.probability = probability
        self.limit = limit

    def __call__(self, x, y, **kwargs):
        dice = self.rng.uniform(0, 1)
        if dice < self.probability:
            factor = 1 + self.rng.uniform(0, self.limit)
            return super().__call__(x, y, factor=factor)
        else:
            return x, y

    def transform_x(self, x, **kwargs):
        factor = kwargs.pop('factor', None)
        w = x.shape[-1]
        new_w = int(factor * w)

        x_t = np.zeros((x.shape[0], new_w))
        for i, row in enumerate(x):
            x_t[i, :] = np.interp(
                np.linspace(0, w - 1, new_w),
                np.linspace(0, w - 1, w),
                row,
            )
        return x_t

    def transform_y(self, y, **kwargs):
        if y is None:
            return y

        factor = kwargs.pop('factor', None)

        y_t = list()
        for mark_tag, mark_vals in y:
            mark_vals = [[int(item[0] * factor), int(item[1] * factor)] for item in mark_vals]

            y_t.append(
                (mark_tag, mark_vals)
            )
        return y_t


class RemoveBordelineMarks:
    def __init__(self, min_width, **kwargs):
        self.min_width = min_width

    def __call__(self, x, y, **kwargs):
        if y is None:
            return x, y

        w = x.shape[-1]
        y_t = list()
        for mark_tag, mark_vals in y:
            temp = list()
            for lower_mark, upper_mark in mark_vals:
                # skip cropped marks
                if lower_mark < self.min_width:
                    continue

                if upper_mark >= w - self.min_width:
                    continue

                if upper_mark - lower_mark < self.min_width:
                    continue
                temp.append([lower_mark, upper_mark])

            y_t.append(
                (mark_tag, sorted(temp))
            )
        return x, y_t


class CheckMarksValidity:
    def __init__(self, **kwargs):
        pass

    def __call__(self, x, y, **kwargs):
        if y is None:
            return x, y

        w = x.shape[-1]
        for mark_tag, mark_vals in y:
            for lower_mark, upper_mark in mark_vals:
                if lower_mark < 0 or upper_mark < 0:
                    raise ValueError(f'Invalid mark value. Expected mark > 0, got {lower_mark} and {upper_mark}.')
                if lower_mark >= w or upper_mark >= w:
                    raise ValueError(f'Invalid mark value. Expected mark < {w}, got {lower_mark} and {upper_mark}.')
                if lower_mark > upper_mark:
                    raise ValueError(f'Invalid mark value. Expected mark[0] to be less than mark[1]')

        return x, y


class RandomChannelSwitch(BaseTransform):
    """
    Class randomly switches two neighboring channels
    """
    def __init__(self, probability: float, use_on: str, **kwargs):
        super().__init__(use_on)
        self.probability = probability

    def __call__(self, x, y, **kwargs):
        dice = self.rng.uniform(0, 1)
        if dice < self.probability:
            return super().__call__(x, y, **kwargs)
        else:
            return x, y

    def transform_x(self, x, **kwargs):
        idx = list(range(0, x.shape[0]))
        ch1 = self.rng.randint(0, x.shape[0]-1)
        ch2 = idx[(ch1 - len(idx)) + 1]
        idx[ch1], idx[ch2] = idx[ch2], idx[ch1]
        return x[idx, :]


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


#----------------------Classes for WaveMap augmentation----------------------
# ---------------------------------------------------------------------------
""" class ZScore:
    Returns Z-score normalized data
    def __init__(self, mean: float = 0, std: float = 1000, **kwargs):
        super().__init__()

        self.mean = mean
        self.std = std

    def __call__(self, x, y, **kwargs):
        return super().__call__(x, y)

    def transform_x(self, x, **kwargs):
        x = x - np.array(self.mean).reshape(-1, 1)
        x = x / self.std

        return x """
    
class ZScoreNormalize:
    def __call__(self, x):
        """
        Args:
            x (Tensor): Shape [channels, num_signals, num_samples]
        
        Returns:
            Normalized Tensor with mean ~0 and std ~1 per signal
        """
        mean = x.mean(dim=-1, keepdim=True)  # Compute mean along num_samples
        std = x.std(dim=-1, keepdim=True)    # Compute std along num_samples
        
        return (x - mean) / (std + 1e-8)  # Normalize, avoid div by zero
    

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
    transform = RandomAmplifier(probability=1, limit=2, shuffle=True, random_seed=42)
    # transform = RandomStretch(probability=0.5, use_on="sample", limit=5)
    x_trans = transform(x)
    #x_trans = transform(x_trans)
    
    
    plot_subplots(x_orig, x_trans, num_subplots=x_orig.shape[0], title="Signal Transform")
    

