import os
import json
import yaml
import h5py
import numpy as np
import pandas as pd
from torch.utils import data
from torch import FloatTensor

import matplotlib.pyplot as plt

#Indication of accessible classes and functions when importing module from outside
__all__ = ["Dataset"]


FIELDS = ['cs9-10', 'cs7-8', 'cs5-6', 'cs3-4', 'cs1-2']

VALID_TERMS = {
    'AVNRT': ('SVT', 1),
    'AVRT': ('SVT', 1),
    'ACJR': ('SVT', 1),
    'NSR': ('NSR', 0),
    'ARAP': ('ARAP', 5),
    'VRAP': ('VRAP', 6),
    'AFL': ('AF', 2),
    'AT': ('AF', 2),
    'AF': ('AF', 2),
    'PVC': ('PVC', 3),
    'PAC': ('PAC', 4),
}

NB_CLASSES = max([item[1] for item in VALID_TERMS.values()]) + 1


def read_yaml(file_name, fields=None):
    """Reads yaml data"""
    with open(file_name, "r") as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return [VALID_TERMS[term['abb']] for term in content['diagnosis'] if term['abb'] in VALID_TERMS]


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def read_csv(file_name, fields=None):
    """Reads csv data"""
    df = pd.read_csv(file_name, sep=',', dtype=np.int32, skipinitialspace=True, usecols=fields)
    return df.transpose().values


def read_h5(file_name, valid_channels: list = None, valid_marks: dict = None):
    if valid_channels is None:
        valid_channels = FIELDS

    with h5py.File(file_name, 'r', libver='latest', swmr=True) as f_obj:
        sampling_f = f_obj.attrs['Fs']

        # get indices of valid channels
        channel_names = [item[0].decode('UTF-8') for item in f_obj['Info']]
        if valid_channels:
            # channel names to indices
            channels_to_read = [channel_names.index(channel) for channel in valid_channels]
            channels_to_read_sorted = sorted(channels_to_read)

            # get relative positions
            relative_idx = [channels_to_read_sorted.index(i) for i in channels_to_read]

            sample = f_obj['Data'][channels_to_read_sorted, :]

            # swap channel as specified by order of elements in `valid_channel`. Cannot be done within reading from h5.
            sample = sample[relative_idx, :]

        else:
            # read entire dataset
            sample = f_obj['Data'][:]

        # read and regroup marks
        if valid_marks:
            marks = {val: [] for val in valid_marks.values()}
            labels = {val: [] for val in valid_marks.values()}

            if 'Marks' in f_obj:
                for mark in f_obj['Marks']:
                    mark_title = mark[-1].decode('UTF-8')

                    if mark_title in valid_marks:
                        key = valid_marks[mark_title]
                        # Store marks
                        marks[key].append(list(mark)[:2])
                        # Store original mark label
                        labels[key].append(
                            remove_prefix(mark_title, prefix='ATRIAL_EGM_CS:')
                            )

                # sort_marks
                formatted_marks, formatted_labels = list(), list()
                for key in marks.keys():
                    try:
                        sorted_marks, sorted_labels = zip(
                            *sorted(
                                zip(marks[key], labels[key]),
                                key=lambda x: x[0][0],
                                )
                            )
                    except ValueError:
                        sorted_marks, sorted_labels = tuple([]), tuple([])
                    formatted_marks.append((key, sorted_marks))
                    formatted_labels.append((key, sorted_labels))
            else:
                marks = None
        else:
            marks = None

    return sample, formatted_marks, formatted_labels, sampling_f


class Dataset(data.Dataset):
    """
    PyTorch Dataset generator class
    Ref: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(
            self,
            filenames_list: list,
            channels: list = None,
            marks: list = None,
            transform=None,
    ):
        """Initialization"""
        self.filenames_list = filenames_list
        self.channels = channels
        self.marks = marks
        self.transform = transform

    def __len__(self):
        """Return total number of data samples"""
        return len(self.filenames_list)

    def __getitem__(self, idx):
        """Generate data sample"""
        # sample_file_name = self.filenames_list[idx]
        sample_file_name = self.filenames_list[idx]
        
        # Read data
        sample, targets, labels, sampling_f = read_h5(
            sample_file_name,
            valid_channels=self.channels,
            valid_marks=self.marks,
        )

        # sample = np.mean(sample, axis=0, keepdims=True)
        # targets = SelParser.read(header_file_name, grouped=False)

        # Transform sample
        if self.transform:
            x, y = self.transform(
                sample,
                targets,
                labels=labels,
                sampling_f=sampling_f,
            )

        return x, y, labels, sample_file_name


class InferenceDataset(data.Dataset):
    def __init__(
            self,
            filenames_list: list,
            channels: list = None,
            transform=None,
    ):
        """Initialization"""
        self.filenames_list = filenames_list
        self.channels = channels
        self.transform = transform

    def __len__(self):
        """Return total number of data samples"""
        return len(self.filenames_list)

    def __getitem__(self, idx):
        """Generate data sample"""
        # sample_file_name = self.filenames_list[idx]
        sample_file_name = self.filenames_list[idx]

        # Read data
        sample, targets, sampling_f = read_h5(
            sample_file_name,
            valid_channels=self.channels,
        )

        # Transform sample
        targets = []
        if self.transform:
            x, y = self.transform(
                sample,
                targets,
                sampling_f=sampling_f,
            )

        x = FloatTensor(x)
        sample_length = FloatTensor([x.shape[1]])

        return x, sample_length, sample_file_name, self.channels





def main():
    pass


if __name__ == "__main__":
    main()
