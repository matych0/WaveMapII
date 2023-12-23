import random
import glob
import json
import os

from itertools import chain
from collections import OrderedDict

import numpy as np

from scipy import stats
from sklearn import cluster
from scipy.cluster.hierarchy import dendrogram
from skmultilearn.model_selection import iterative_train_test_split as iterative_split
from skmultilearn.model_selection import IterativeStratification

import matplotlib.pyplot as plt


def create_label_summary(file_list):
    """
    Function creates a dictionary consisted of a label summary for each file in the dataset folder
    Dictionary is stored as json file. It is supposed to be used for iterative stratified sampling.
    :param file_list:
    :return:
    """

    summary = dict()
    by_file = list()

    idx = 0
    for file in file_list:
        file_id = os.path.basename(file)[0:-4]
        file_path = os.path.join(os.path.dirname(file), file_id + '.json')

        with open(file_path, 'r') as f:
            stats_file = json.load(f)

        temp = list()
        for diagnosis in stats_file['diagnosis']:
            abb = diagnosis['abbreviation']
            temp.append(abb)

            if abb not in summary:
                summary[abb] = idx
                idx += 1

        by_file.append(temp)

    one_hot_labels = np.zeros((len(file_list), len(summary)))

    for idx, item in enumerate(by_file):
        for abb in item:
            one_hot_labels[idx, summary[abb]] = 1

    return one_hot_labels, summary


class BasePartition:
    partition_names = ('train', 'evaluation')

    def __init__(
            self,
            seed: int,
            start: int = 0,
            max_iter: int = 1
    ) -> None:
        self.seed = seed
        self.iteration = start
        self.max_iter = max_iter

    def _shuffle(self, files):
        state = random.getstate()

        # Initialize custom seed
        if self.seed:
            random.seed(self.seed)

        # Shuffle list of file names
        random.shuffle(files)

        # Restore generator state
        random.setstate(state)

        return files

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class HoldoutPartition(BasePartition):
    def __init__(self,
                 files: list,
                 evaluation: float,
                 holdout: float,
                 seed: int,
                 stratified_by: str = None,
                 ) -> None:

        # Check validity of splitting ratios
        if not 0 <= evaluation < 1:
            raise ValueError(f'Relative size of evaluation dataset is out of range')

        if not 0 <= holdout < 1:
            raise ValueError(f'Relative size of holdout dataset is out of range')

        if not holdout + evaluation < 1:
            raise ValueError(f'Cumulative size of holdout and evaluation dataset is out of range')

        super().__init__(seed=seed)

        self.left = int(len(files) * holdout)
        self.right = int(len(files) * evaluation)

        files = files.copy()
        if stratified_by == 'parent_folder':
            self.files = self.stratify_folders(files, evaluation)
            print()
            # holdout_part = self._shuffle(sorted(files[0:self.left]))
            # eval_part = self._shuffle(sorted(files[self.left:self.left + self.right]))
            # train_part = self._shuffle(sorted(files[self.left + self.right::]))
            # self.files = train_part + eval_part
        else:
            self.files = self._shuffle(files)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration <= self.max_iter:
            self.iteration += 1
            # hold_out = self.files[0:self.left]

            # output: (train, evaluation). Train expected at the end of list.
            return (
                self.files[self.left + self.right::],
                self.files[self.left:self.left + self.right],
            )

        else:
            raise StopIteration

    def stratify_folders(self, files, evaluation) -> list:

        shuffled_files = self._shuffle(files)
        folders = [os.path.basename(os.path.dirname(item)) for item in shuffled_files]

        unique_folders = OrderedDict()

        for i, file in enumerate(shuffled_files):
            folder = os.path.basename(os.path.dirname(file))

            if folder not in unique_folders:
                unique_folders.update({folder: []})

            unique_folders[folder].append(i)

        counter = 0
        train_files, eval_files = [], []

        for folder in unique_folders:
            for file_idx in unique_folders[folder]:
                if counter < int(len(files) * (1 - evaluation)):
                    train_files.append(shuffled_files[file_idx])
                else:
                    eval_files.append(shuffled_files[file_idx])
                counter += 1

        return eval_files + train_files


def stratified_partition(dataset, evaluation_size=0.3, holdout_size=0.05, seed=None):
    """
    Iteratively partitions the dataset into train, evaluation and test subset with balanced distribution.

    :param dataset: list of file names
    :param labels: list of file names
    :param evaluation_size: float: relative size of evaluation subset
    :param holdout_size: float: relative size of holdout subset
    :return: dict: subsets of file names
    """

    label_summary, summary = create_label_summary(dataset)

    k_fold = IterativeStratification(n_splits=2, order=2, random_state=seed)

    train, test = k_fold.split(np.expand_dims(np.arange(start=0, stop=len(dataset)), 1), label_summary)

    x_train, y_train, x_test, y_test = iterative_split(
        np.expand_dims(np.arange(start=0, stop=len(dataset)), 1),
        label_summary,
        test_size=evaluation_size,
    )

    x_train = np.squeeze(x_train, axis=1)
    x_test = np.squeeze(x_test, axis=1)

    partition = {
        'holdout': [],
        'evaluation': [],
        'train': [],
    }

    for idx in x_train:
        partition['train'].append(dataset[idx])

    for idx in x_test:
        partition['evaluation'].append(dataset[idx])

    return partition

