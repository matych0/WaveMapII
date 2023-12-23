import math
from abc import ABCMeta
from typing import Union, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def matthews(tp, fp, tn, fn):
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator > 0:            
        return ((tp * tn) - (fp * fn)) / np.sqrt(denominator)                
    else:
        return float('nan')


def f_score(tp, fp, tn, fn):    
    beta=1
    if beta * tp + fp + fn:        
        return ((1.0 + beta ** 2) * tp) / ((1.0 + beta ** 2) * tp + fp + beta ** 2 * fn)
    else:
        return float('nan')


def recall(tp, fp, tn, fn):
    if tp + fn > 0:        
        return tp / (tp + fn)
    else:
        return float('nan')


def precision(tp, fp, tn, fn):
    if tp + fp > 0:        
        return tp / (tp + fp)
    else:
        return float('nan')


def accuracy(tp, fp, tn, fn):
    if tp + fp + tn + fn > 0:        
        return (tp + tn) / (tp + fp + tn + fn)
    else:
        return float('nan')


def nearest_value(value, arr):
    """
    Searches for the position of the nearest value in the array <arr>.
    """
    if arr.size == 0:
        return None, 2147483647

    idx = np.searchsorted(arr, value, side="left")
    if idx == len(arr) or math.fabs(value - arr[idx - 1]) < math.fabs(value - arr[idx]):
        idx -= 1
    return idx, arr[idx]


function_dict = {
    'matthews': matthews,
    'f_score': f_score,
    'recall': recall,
    'precision': precision,
}


class MetricMetaClass(metaclass=ABCMeta):
    def __init__(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def compute(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass


class ScoreTracker(MetricMetaClass):
    def __init__(self):
        super().__init__()
        self.score = np.nan
    
    def reset(self):
        self.score = 0

    def update(self, value):
        self.score = value


class AvgLoss(MetricMetaClass):
    """Class representing average loss across entire dataset."""
    def __init__(self):
        super().__init__()
        self.sum = 0
        self.nb_samples = 0

    def reset(self):
        self.sum = 0
        self.nb_samples = 0

    def update(self, avg_loss, nb_samples):
        """
        Aggregates prediction errors using specific loss function.
        :param y_pred: (float) raw prediction scores
        :param targets: (float) one-hot-encoded targets
        :return: None
        """

        # Aggregate statistics
        self.sum += avg_loss * nb_samples
        self.nb_samples += nb_samples

    def compute(self):        
        return self.sum / self.nb_samples
        

class ConfusionMatrix(MetricMetaClass):
    """Class representing confusion matrix."""
    def __init__(self, nb_classes: int, normalize: bool = False, limit=0.5):
        super().__init__()
        if isinstance(limit, list):
            if len(limit) != nb_classes:
                raise ValueError('number of elements in a list of "thresholds" must be equal to "nb_classes"')
        self.nb_classes = nb_classes
        self.limit = limit
        self.normalize = normalize
        self._cfm = np.zeros((self.nb_classes, 4))

    def reset(self):
        self._cfm = np.zeros((self.nb_classes, 4))

    def update(self, y_pred, targets, mask=None):
        """
        Aggregates absolute/relative counts of true positive, true negative, false positive
        and false negative classifications.
        :param y_pred: (float) one-hot-encoded predictions
        :param targets: (float) one-hot-encoded targets
        :param mask: (float) masking of zero-padded parts
        """

        if mask is None:
            mask = np.ones(targets.shape)
        y_pred = y_pred * mask

        for local_y, local_targets, local_mask in zip(y_pred, targets, mask):
            # row normalization for multi-label problem.
            norm_factor = float(max(np.sum(local_y), 1)) if self.normalize else 1.0

            for class_idx in range(self.nb_classes):

                limit = self.limit[class_idx] if isinstance(self.limit, list) else self.limit

                # TP
                self._cfm[class_idx, 0] += int(np.count_nonzero(
                    np.bitwise_and(
                        local_y >= limit,
                        local_targets == 1),
                )
                ) / norm_factor

                # FP
                self._cfm[class_idx, 1] += np.count_nonzero(
                    np.bitwise_and(
                        local_y >= limit,
                        local_targets == 0,
                    )
                ) / norm_factor

                # TN
                self._cfm[class_idx, 2] += np.count_nonzero(
                    np.bitwise_and(
                        local_y < limit,
                        local_targets == 0,
                    )
                ) / norm_factor

                # FN
                self._cfm[class_idx, 3] += np.count_nonzero(
                    np.bitwise_and(
                        local_y < limit,
                        local_targets == 1,
                    )
                ) / norm_factor

    def compute(self):
        """Returns confusion matrix"""
        return self._cfm


class StatsWarehouse:
    def __init__(
            self,            
            cols: Union[None, tuple],
            formatting: Union[None, tuple] = None,
            delimiter: str = ' | ',
            print_cols: Union[list, tuple, None] = None,

    ):  
        self.cols = cols
        self.df = pd.DataFrame(columns=self.cols)
        self.formatting = formatting
        self.delimiter = delimiter
        self._fmts = {
            col_name: (lambda fmt: (lambda x: f'{x:{fmt}}'))(fmt) for col_name, fmt in zip(self.cols, self.formatting) if fmt is not None
            }

        self._print_cols = set(print_cols) if print_cols is not None else None
    
    def reset(self):
        self.df = pd.DataFrame(columns=self.cols)

    def update(self, statistics: list):
        df = pd.DataFrame([statistics], columns=self.cols)        

        # append to current df
        self.df = pd.concat([self.df, df], ignore_index=True)

    def store(self, path: str):
        df = self.df.copy()

        for col_name, fmt in self._fmts.items():            
            df[col_name] = df[col_name].apply(fmt)
        df.to_csv(path, index=False, sep='\t')

    def stringify(self, row_idx: int, delimiter: Union[str, None]=None):
        """Returns formatted row

        Args:
            row_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if delimiter is None:
            delimiter = self.delimiter

        formatted_values = []
        for fmt, (col_name, value) in zip(self.formatting, self.df.iloc[row_idx].iteritems()):
            if self._print_cols is not None:
                if col_name in self._print_cols:
                    try:
                        formatted_values.append(f'{col_name}: {value:{fmt}}')
                    except ValueError:
                        formatted_values.append(f'{col_name}: {value}')

        return delimiter.join(formatted_values)
    

class RegressionMetrics:
    def __init__(
            self,
            limit: int,
            cols: Union[None, tuple] = ('estimate', 'reference', 'id', 'group'),
            unit_scale: float = 1,
    ):
        self.limit = limit
        self.cols = cols
        self.df = pd.DataFrame(columns=self.cols)
        self.missing_value = float('NaN')
        self.unit_scale = unit_scale

        self._counter = 0
        self._true_positive = 0
        self._false_positive = 0
        self._false_negative = 0

    def reset(self):
        self.df = pd.DataFrame(columns=self.cols)

        self._counter = 0
        self._true_positive = 0
        self._false_positive = 0
        self._false_negative = 0

    def update(
            self,
            y: Union[np.array, List, Tuple],
            targets: Union[np.array, List, Tuple],
            **kwargs,
    ):
        oid = kwargs.pop('oid', self.missing_value)
        groups = kwargs.pop('groups', [self.missing_value] * len(y))    
        groups = list(groups)

        true_positive, false_positive, false_negative = 0, 0, 0

        if isinstance(y, (list, tuple)):
            y = np.array(y)

        if isinstance(targets, (list, tuple)):
            targets = np.array(targets)

        # rescale to arbitrary units
        y = y * self.unit_scale
        targets = targets * self.unit_scale

        pairings = list()
        for value in y:
            target_idx, target_value = nearest_value(value, targets)

            if (target_value - self.limit) <= value <= (target_value + self.limit):
                # true positives
                true_positive += 1
                group = groups[target_idx]

                # remove target_value due to possible future duplicates
                targets = np.delete(targets, target_idx)
                groups.pop(target_idx)
            else:
                # false positives
                false_positive += 1
                target_value = self.missing_value
                group = self.missing_value
            
            pairings.append([value, target_value, oid, group])

        if targets.size > 0:
            for target_value, group in zip(targets, groups):
                # false negatives
                false_negative += 1
                pairings.append([self.missing_value, target_value, oid, group])

        df = pd.DataFrame(pairings, columns=self.cols)

        self._true_positive += true_positive
        self._false_positive += false_positive
        self._false_negative += false_negative

        # append to current df
        self.df = pd.concat([self.df, df], ignore_index=True)

        self._counter += 1

    def store(self, path: str):
        df = self.df.copy()
        df.to_csv(path, index=False, sep='\t', na_rep='None')

    def precision(self):
        if self._true_positive + self._false_positive > 0:
            return self._true_positive / (self._true_positive + self._false_positive)
        else:
            return float('nan')

    def recall(self):
        if self._true_positive + self._false_negative > 0:
            return self._true_positive / (self._true_positive + self._false_negative)
        else:
            return float('nan')

    def rmse(self):
        df = self.df[self.df['estimate'].notna() & self.df['reference'].notna()].copy()
        if len(df) > 0:
            return np.sqrt(np.sum((df['estimate'] - df['reference']) ** 2) / len(df))
        else:
            return float('NaN')

    def rsquared(self):
        df = self.df[self.df['estimate'].notna() & self.df['reference'].notna()].copy()
        if len(df) > 0:
            return r2_score(df['reference'], df['estimate'])
        else:
            return float('NaN')


class PerFileWarehouse:
    def __init__(
            self,
            limit: float,
            cols: Union[None, tuple] = ('id', 'group', 'TP', 'FP', 'TN', 'FN'),
    ):
        self.limit = limit
        self.cols = cols
        self.df = pd.DataFrame(columns=self.cols)
        self.missing_value = float('NaN')

        self._counter = 0

    def reset(self):
        self.df = pd.DataFrame(columns=self.cols)

        self._counter = 0

    def update(
            self,
            y: Union[np.array, List, Tuple],
            targets: Union[np.array, List, Tuple],
            mask=None,
            **kwargs,
    ):
        oid = kwargs.pop('oid', self.missing_value)
        groups = kwargs.pop('groups')

        if mask is None:
            mask = np.ones(targets.shape)
        y = y * mask

        pairings = list()

        # last valid sample in padded sequence
        max_idx = np.max(np.nonzero(mask))

        # TP
        tp, fp, tn, fn = 0, 0, 0, 0
        tp = int(np.count_nonzero(
            np.bitwise_and(
                y[:max_idx] >= self.limit,
                targets[:max_idx] == 1),
            )
        )

        # FP
        fp = int(np.count_nonzero(
            np.bitwise_and(
                y[:max_idx] >= self.limit,
                targets[:max_idx] == 0),
            )
        )

        # TN
        tn = int(np.count_nonzero(
            np.bitwise_and(
                y[:max_idx] < self.limit,
                targets[:max_idx] == 0),
            )
        )

        # FN
        fn = int(np.count_nonzero(
            np.bitwise_and(
                y[:max_idx] < self.limit,
                targets[:max_idx] == 1),
            )
        )

        pairings.append([oid, groups, tp, fp, tn, fn])

        df = pd.DataFrame(pairings, columns=self.cols)

        # append to current df
        self.df = pd.concat([self.df, df], ignore_index=True)

        self._counter += 1

    def store(self, path: str):
        df = self.df.copy()
        df.to_csv(path, index=False, sep='\t', na_rep='None')


class BinaryMetrics(MetricMetaClass):
    """
    Methews correlation coefficient
    """
    def __init__(self, cfm: np.array, metrics: list):
        """
        :param beta: Metric custom coefficient
        """
        super().__init__()        
        self.cfm = cfm

        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]

        self.metrics = metrics        

    def compute(self, partial=False):
        scores = {key: [] for key in self.metrics}

        # iterate over each class
        for tp, fp, tn, fn in self.cfm.compute():
            for stat in self.metrics:
                partial_score = function_dict[stat](tp, fp, tn, fn)
                
                # set nans to zero
                if np.isnan(partial_score):
                    partial_score = np.nan
                
                scores[stat].append(partial_score)

        if not partial:
            scores = {key: np.nanmean(value) for key, value in scores.items()}
        
        return scores  

    def reset(self):
        pass


class PointwiseConfusionMatrix(ConfusionMatrix):
    def __init__(self, nb_classes: int, limit: float, normalize: bool = False):
        """_summary_

        Args:
            nb_classes (int): _description_
            limit (float): Accepting deviation between prediction and reference in samples?
            normalize (bool, optional): _description_. Defaults to False.
        """
        super().__init__(nb_classes, normalize, limit)

    def arrify(self, targets):
        s = np.array([i for i, _ in targets], dtype=float)
        e = np.array([i for _, i in targets], dtype=float)
        m = s + ((e - s) / 2)        

        return s, e, m


    def search_nearest(self, value, arr):
        idx = np.searchsorted(arr, value, side="left")

        if idx == len(arr) or math.fabs(value - arr[idx - 1]) < math.fabs(value - arr[idx]):        
            idx -= 1
        
        return idx, arr[idx]


    def update(self, y, targets):
        # single class only !!!
        class_idx = 0

        for y_row, t_row in zip(y, targets):
            # no targets, some prediction
            if not t_row:
                FP = len(y_row)
                self._cfm[:, 1] += FP
                continue

            # no prediction, some targets
            if not t_row:
                FN = len(targets)
                self._cfm[:, 3] += FN
                continue

            # arrify target into start, end and middle positions
            s, e, m = self.arrify(t_row)
            
            for s_mark, e_mark in y_row:                
                m_mark = s_mark + ((e_mark - s_mark) / 2)
                
                for idx, mark, arr in zip(
                    range(3),
                    (s_mark, e_mark, m_mark),
                    (s, e, m),
                ):
                    t_ind, t_v = self.search_nearest(mark, s)

                    if mark >= (t_v - self.limit) and mark <= (t_v + self.limit):
                        self._cfm[idx, 0] += 1 #TP
                        arr[t_ind] = np.Inf
                    else:
                        self._cfm[idx, 1] += 1 #FP

            self._cfm[0, 3] += np.count_nonzero(np.isfinite(s))  #FN
            self._cfm[1, 3] += np.count_nonzero(np.isfinite(e))  #FN
            self._cfm[2, 3] += np.count_nonzero(np.isfinite(m))  #FN


class ConfusionMatrixStats(MetricMetaClass):
    """
    Methews correlation coefficient
    """
    def __init__(self, cfm: np.array, stats: list, params: dict = None):
        """
        :param beta: Metric custom coefficient
        """
        super().__init__()        
        self.cfm = cfm
        self.stats = stats
        self.params = params

    def compute(self, partial=False):
        scores = {key: [] for key in self.stats}

        # iterate over each class
        for tp, fp, tn, fn in self.cfm.compute():
            for stat in self.stats:
                partial_score = function_dict[stat](tp, fp, tn, fn, self.params)
                
                # set nans to zero
                if np.isnan(partial_score):
                    partial_score = np.nan
                
                scores[stat].append(partial_score)

        if not partial:
            scores = {key: np.nanmean(value) for key, value in scores.items()}
        
        return scores  

    def reset(self):
        pass






