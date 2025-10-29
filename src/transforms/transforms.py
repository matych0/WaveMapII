import numpy as np
from scipy.signal import resample


class BaseTransform:
    """ Base class for all transforms """
    def __init__(self, shuffle=False, random_seed=None, **kwargs):
        self.shuffle = shuffle  
        self.np_rng = np.random.default_rng(random_seed)

    def __call__(self, x, **kwargs):
        if self.shuffle:
            self.np_rng.shuffle(x)     
        return self.transform(x, **kwargs)

    def transform(self, x, **kwargs):
        return x


# -------------------- Classes for data normalization --------------------
# ------------------------------------------------------------------------


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
    
    
class TanhNormalize(BaseTransform):
    """Returns tanh normalized data"""
    def __init__(self, factor: float = 0, **kwargs):
        super().__init__()

        self.factor = factor

    def __call__(self, x, **kwargs):
        return super().__call__(x)

    def transform(self, x, **kwargs):
        x = np.tanh(x / self.factor)

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
    def __init__(self, probability: float, shift_range: float = 0.9, arbitrary: float = None, shuffle: bool = True, random_seed: int = None, **kwargs):
        assert 0.0 < shift_range <= 1.0, "shift_range should be in (0, 1]."
        
        self.probability = probability
        self.shift_range = shift_range
        self.arbitrary = arbitrary
        self.rng = np.random.default_rng(random_seed)
        super().__init__(shuffle=shuffle, random_seed=random_seed)


    def __call__(self, x, **kwargs):
        return super().__call__(x)

    def transform(self, x, **kwargs):

        num_signals, signal_length = x.shape
        num_transformed = round(self.probability * num_signals)  # Number of signals to shift

        # Generate random shift fractions in range [-shift_range, shift_range]
        shift_fractions = self.rng.uniform(-self.shift_range, self.shift_range, size=num_transformed)

        if self.arbitrary is not None:
            shift_fractions = np.array([self.arbitrary] * num_transformed)

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
    def __init__(self, probability: float, low_limit: float, high_limit: float, arbitrary: float = None, shuffle: bool = True, random_seed: int = None, **kwargs):
        super().__init__(shuffle=shuffle, random_seed=random_seed)
        self.probability = probability
        self.rng = np.random.default_rng(random_seed)
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.arbitrary = arbitrary
        

    def __call__(self, x, **kwargs):
        return super().__call__(x)

    def transform(self, x, **kwargs):        
        
        num_signals, signal_length = x.shape
        num_transformed = round(self.probability * num_signals)  # Number of signals to add noise to

        noise_dbs = self.rng.uniform(self.low_limit, self.high_limit, size=num_transformed)

        if self.arbitrary is not None:
            noise_dbs = np.array([self.arbitrary] * num_transformed)

        for i, db in zip(range(num_transformed), noise_dbs):
            # estimate SNR
            power = np.sqrt(np.mean(x[i, :] ** 2) / (10 ** (db / 10)))
            # add noise
            x[i] += self.rng.normal(loc=0, scale=power, size=signal_length)
        return x


class RandomAmplifier(BaseTransform):
    """ Class randomly amplifies signal """
    def __init__(self, probability: float, limit: float, arbitrary: float = None, shuffle: bool = True, random_seed: int = None, **kwargs):
        super().__init__(shuffle=shuffle, random_seed=random_seed)

        self.rng = np.random.default_rng(random_seed)
        self.probability = probability
        self.limit = limit
        self.arbitrary = arbitrary

    def __call__(self, x, **kwargs):
        return super().__call__(x, **kwargs)


    def transform(self, x, **kwargs):

        num_signals, signal_length = x.shape
        num_transformed = round(self.probability * num_signals)  # Number of signals to amplify

        amplify_factors = self.rng.uniform(1 - self.limit, 1+ self.limit, size=num_transformed)
        
        if self.arbitrary is not None:
            amplify_factors = np.array([self.arbitrary] * num_transformed)

        x[:num_transformed] *= amplify_factors[:, np.newaxis]

        return x


class RandomTemporalScale(BaseTransform):
    """
    Class randomly stretches temporal dimension of signal
    """
    def __init__(self, probability: float, limit: float, arbitrary: float = None, shuffle: bool = True, random_seed: int = None, **kwargs):
        super().__init__(shuffle=shuffle, random_seed=random_seed)

        self.rng = np.random.default_rng(random_seed)
        self.probability = probability
        self.limit = limit
        self.arbitrary = arbitrary

    def __call__(self, x, **kwargs):
        return super().__call__(x, **kwargs)

    def transform(self, x, **kwargs):
        num_signals, width = x.shape
        num_transformed = round(self.probability * num_signals)  # Number of signals to amplify

        factors = self.rng.uniform(1 - self.limit,1 + self.limit, size=num_transformed)

        if self.arbitrary is not None:
            factors = np.array([self.arbitrary] * num_transformed)

        new_widths = (factors * width).astype(int)

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
        x_resampled[num_transformed:] = x[num_transformed:]

        return x_resampled