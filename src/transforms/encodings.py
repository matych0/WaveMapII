import random
import itertools
from abc import ABC, abstractmethod
import numpy as np
from scipy import signal
from scipy.signal import firwin, filtfilt, butter
import json
from matplotlib import pyplot as plt

__all__ = ["Compose", "HardClip", "ZScore", "RandomShift", "RandomStretch", "RandomAmplifier", "RandomLeadSwitch",
           "Resample", "BaseLineFilter", "OneHotEncoding", "AddEmgNoise"
           ]


def resample(y:list, factor: float) -> list:
    """ Rescale positions of the marks when resampling of the signal is required.

    Args:
        y (list): List with time marks
        factor (float): Scaling factor

    Returns:
        list: List with resampled time marks
    """
    y_t = list()
    for batch in y:
        temp = list()
        for left_mark, righ_mark in batch:
            temp.append([left_mark * factor, righ_mark * factor])
        y_t.append(temp)
    return y_t


def to_onehot(targets: list, max_len: int) -> np.array:
    """ Converts time marks to binary sequence.

    Args:
        targets (list): List with time marks
        max_len (int): The length of the longest signal in the batch

    Returns:
        np.array: one-hot encoded marks
    """
    y_t = np.zeros([len(targets), max_len])
    for i, intervals in enumerate(targets):
        for lower_mark, upper_mark in intervals:
            lower_mark = int(lower_mark // 1)
            upper_mark = int(-(upper_mark // -1))

            y_t[i, lower_mark:upper_mark] = 1.0
    return y_t


def extend_instances(y: list, size: int) -> list:
    """ Creates margin for each time mark.

    Args:
        y (list): List with time marks
        size (int): Margin from the marks (to the both sides left and right)

    Returns:
        list: List of margins
    """
    y_t = list()
    for instance in y:
        temp = list()
        for value in instance:
            left, right = value - size, value + size
            left = max(0, left)
            temp.append([left, right])
        y_t.append(temp)
    return y_t


def center_instances(y: list) -> list:
    y_t = list()
    for instance in y:
        temp = list()
        for left, right in instance:
            center = left + ((right - left) // 2)
            temp.append(center)
        y_t.append(temp)
    return y_t


def to_instances(y: np.array, threshold: float) -> list:
    """ Decodes a sequence of sample-wise predictions into individual instances.
    Only for single class output.

    Args:
        y (np.array): Model's normalized scores 
        threshold (float): Decision threshold

    Raises:
        ValueError: Check for the consistency in detected number of onset and offset pairs.

    Returns:
        list: List of marks.
    """
    # expand to 2d for compatibility with batched inputs
    if y.ndim == 1:
        y = np.expand_dims(y, 0)

    temp = np.where(y >= threshold, 1, 0)
    temp = np.pad(temp, ((0, 0), (1, 1)), 'constant')
    delta = np.diff(temp, axis=-1)

    rowup, colup = np.nonzero(delta == 1)
    rowdow, coldow = np.nonzero(delta == -1)

    if not all(rowup == rowdow):
        raise ValueError(f'Indexes of rows for given marks does not match. Got {rowup} != {rowdow}.')

    if not all(coldow > colup):
        raise ValueError(f'Invalid values of marks does not match. Got mark_right < mark_left.')
    # preallocate nested list for each row
    marks = [[] for _ in range(y.shape[0])]
    for idx, left, right in zip(rowup, colup, coldow):
        marks[idx].append([left, right])

    return marks