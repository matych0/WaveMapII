import torch
import numpy as np
import random


class BaseCollate:
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, batch):
        return self.collate(batch)

    def collate(self, batch):
        raise NotImplementedError


class PaddedCollate(BaseCollate):
    """
    Create padded mini-batch of training samples along dimension dim
    """
    def __init__(self, dim: int, val: float):
        super(PaddedCollate, self).__init__(dim=dim)
        self.padval = val

    def collate(self, batch):
        """
        Returns padded mini-batch
        :param batch: (list of tuples): tensor, label
        :return: padded_array - a tensor of all examples in 'batch' after padding
        labels - a LongTensor of all labels in batch
        sample_lengths – origin lengths of input data
        """
        mask, widths = None, None
        batch_size = len(batch)
        in_channels = batch[0][0].shape[0]

        widths = [sample[0].shape[self.dim] for sample in batch]

        # find the longest sequence
        x = self.padval * np.ones((batch_size, in_channels, max(widths)), dtype=np.float32)

        # preallocate padded NumPy array
        files, target, labels = list(), list(), list()

        # fill out preallocated padded tensors by data samples and target one-hot encoded labels
        for idx, sample in enumerate(batch):
            x[idx, :, :widths[idx]] = sample[0]

            if sample[1] is None:
                target.append([])
            else:
                target.append(sample[1][0][1])

            labels.append(sample[2])
            files.append(sample[3])

        # Pass to Torch Tensor
        x = torch.from_numpy(x).float()

        return x, target, widths, mask, labels, files
