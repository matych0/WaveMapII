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
    

# -------------------- Class for patch processing --------------------
# ------------------------------------------------------------------------

class ShufflePatch:
    def __init__(self, random_seed=None):
        self.rng = np.random.default_rng(random_seed)

    def __call__(self, x):
        B, P = x.shape[:2]
        perms = np.argsort(self.rng.random((B, P)), axis=1)
        perms = perms.reshape(B, P, *([1] * (x.ndim - 2)))
        return np.take_along_axis(x, perms, axis=1)

    
class ZScorePerSignal:
    """
    Z-score normalization applied independently to every signal
    along the last dimension.

    Works with arrays shaped (..., signal_length)
    Example: (200, 24, 203)
    """

    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + self.eps)
    

class RandomSignalZeroing:
    def __init__(self, max_prob=0.2, random_seed=None):
        self.max_prob = max_prob
        self.rng = np.random.default_rng(random_seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("Input must have shape (patches, channels, length)")

        P, C, L = x.shape
        total_signals = P * C

        # draw probability
        p = self.rng.uniform(0.0, self.max_prob)

        # number of signals to zero
        n_zero = int(round(p * total_signals))
        if n_zero == 0:
            return x

        # choose random signal indices
        flat_indices = self.rng.choice(total_signals, size=n_zero, replace=False)

        # convert flat -> (patch, channel)
        patch_idx = flat_indices // C
        channel_idx = flat_indices % C

        # zero entire signals
        x[patch_idx, channel_idx, :] = 0.0

        return x


class RandomSignalFlipping:
    def __init__(self, max_prob=0.2, random_seed=None):
        self.max_prob = max_prob
        self.rng = np.random.default_rng(random_seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:

        if x.ndim != 3:
            raise ValueError("Input must have shape (patches, channels, length)")

        P, C, L = x.shape
        total_signals = P * C

        # draw probability
        p = self.rng.uniform(0.0, self.max_prob)

        # number of signals to zero
        n_zero = int(round(p * total_signals))
        if n_zero == 0:
            return x

        # choose random signal indices
        flat_indices = self.rng.choice(total_signals, size=n_zero, replace=False)

        # convert flat -> (patch, channel)
        patch_idx = flat_indices // C
        channel_idx = flat_indices % C

        # flip polarity of entire signals
        x[patch_idx, channel_idx, :] *= -1.0

        return x
    

class RandomPatchFlipping:
    def __init__(
        self, 
        max_prob=0.2, 
        random_seed=None
    ):
        
        self.max_prob = max_prob
        self.rng = np.random.default_rng(random_seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("Input must have shape (P, C, L)")

        P, C, L = x.shape

        # draw probability of patch selection
        p = self.rng.uniform(0.0, self.max_prob)
        n_flip = int(round(p * P))

        if n_flip == 0:
            return x

        # select patches
        patch_indices = self.rng.choice(P, size=n_flip, replace=False)

        # flip polarity of selected patches
        x[patch_indices] *= -1.0

        return x
    

class RandomPatchTimeShift:
    def __init__(
        self,
        max_prob=0.2,
        max_shift_frac=0.1,
        random_seed=None,
    ):
        self.max_prob = max_prob
        self.max_shift_frac = max_shift_frac
        self.rng = np.random.default_rng(random_seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ValueError("Input must have shape (P, C, L)")

        P, C, L = x.shape

        # draw probability of patch selection
        p = self.rng.uniform(0.0, self.max_prob)
        n_shift = int(round(p * P))

        if n_shift == 0:
            return x

        # select patches
        patch_indices = self.rng.choice(P, size=n_shift, replace=False)


        max_shift = int(round(self.max_shift_frac * L))
        if max_shift == 0:
            return x

        for pidx in patch_indices:
            shift = self.rng.integers(-max_shift, max_shift + 1)
            if shift == 0:
                continue

            x[pidx] = np.roll(x[pidx], shift, axis=-1)

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