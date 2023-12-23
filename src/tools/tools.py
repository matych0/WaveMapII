import os
import yaml
import random
import numpy as np
import torch
import h5py

from copy import deepcopy

from statsmodels.stats.gof import chisquare
from scipy.special import kl_div, rel_entr


class BestModel:
    def __init__(self, model):
        super().__init__()
        
        self.best_score = float('-Inf')
        self.model = model
        self.best_model = None   
        self.is_updated = False

    def update(self, score, **kwargs):
        if score > self.best_score:
            self.best_score = score

            # deepcopy model
            self.best_model = deepcopy(self.model).to('cpu')

            # switch to evaluation mode
            self.best_model.eval()            
            
            self.is_updated = True
        else:
            self.is_updated = False

    def store(self, path: str, **kwargs):

        suffix = kwargs.pop('suffix', '')
        storage_path = os.path.join(path, self.model._name + '_' + self.model._version + suffix + '.pt')
        
        # export model to TorchScript and save to hdd
        if self.best_model:
            torch.save(
                self.best_model.state_dict(),
                storage_path,
            ) 


class SelParser:
    def __init__(self):
        pass

    @staticmethod
    def read(f_name: str, valid_labels: str = None):
        """Returns dict containing sample information"""    

        if isinstance(valid_labels, str):
            valid_labels = {valid_labels}

        label_dict = dict()
        with open(f_name, 'r') as file:
            for row in file:
                # skip header
                if row[0] == '%':
                    continue

                row = row.rstrip('\n')

                # skip empty lines
                if not row:
                    continue

                # parse single line
                prow = row.split('\t')
                try:
                    idx, left_mark, right_mark, group, _, _, _, label = prow
                except ValueError:
                    print()

                if valid_labels is not None:
                    if label not in valid_labels:
                        continue

                if label not in label_dict:
                    label_dict[label] = []
                label_dict[label].append([int(left_mark), int(right_mark)])

        return label_dict

    def write(file_name):
        """Write .sel file"""
        # TO DO

        
def output_to_sel(y, file_list):
    ch_names = ''.join(['%' + item + '\t1\n' for item in ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'cs1-2', 'cs3-4', 'cs5-6', 'cs7-8', 'cs9-10', 'his1-2', 'his3-4', 'stim')])

    one_hot = np.where(y >= 0.5, 1, 0)
    for i, file_name in enumerate(file_list):
        export_name = file_name[:-3] + '.sel'

        temp = one_hot[i, :, :]
        temp = np.pad(temp, (1, 1), 'constant')
        delta = np.diff(temp)

        output_txt = ''
        idx = 0
        for row_idx, row in enumerate(delta):
            up_div = np.flatnonzero(row == 1)
            down_div = np.flatnonzero(row == -1)

            for start, stop in zip(up_div, down_div):
                output_txt += f'{idx}\t{4 * start}\t{4 * stop}\t{0}\t{1.0}\t{12}\t{"cs1-2"}\t{"DNN_C-"}{row_idx}\n'
                idx += 1

        text = (
            '%SignalPlant ver.:1.2.7.3\n'
            '%Selection export from file:\n'
            f'%{os.path.basename(file_name)}\n'
            f'%SAMPLING_FREQ [Hz]:{2000}\n'
            '%CHANNELS_VALIDITY-----------------------\n'
            f'{ch_names}'
            '%----------------------------------------\n'
            '%Structure:\n'
            '%Index[-], Start[sample], End[sample], Group[-], Validity[-], Channel Index[-], Channel name[string], Info[string]\n'
            '%Divided by: ASCII char no. 9\n'
            '%DATA------------------------------------\n'
            f'{output_txt}\n')

        with open(export_name, 'w') as f_obj:
            f_obj.writelines(text)


def compute_weights(valid_targets, nb_files):
    w_positive = [0] * len(valid_targets)
    w_negative = [0] * len(valid_targets)

    for item in valid_targets.values():
        w_positive[item['index']] = nb_files / item['counts'] if item['counts'] > 0 else 0
        w_negative[item['index']] = nb_files / (nb_files - item['counts']) if item['counts'] > 0 else 0

    return w_positive, w_negative


def test_partion(dset, valid):
    names, counts, total_counts = dict(), dict(), dict()
    
    for idx, (_, subset) in enumerate(dset.items()):
        for file_path in subset:
            file_path = file_path[:-3] + '.yaml'
            with open(file_path, "r") as stream:
                ymlf = yaml.safe_load(stream)
            # count diagnosis
            for item in ymlf['diagnosis']:
                abb = str(item['abb'])

                if abb not in valid:
                    continue

                if abb not in counts:
                    counts[abb] = [0, 0]
                                                    
                counts[abb][idx] += 1
                
    t = [item[0] for item in counts.values()]
    e = [item[1] for item in counts.values()]

    ratio = np.sum(t) / np.sum(e)
    corected_e = [int(ratio * item) for item in e]
    chi = chisquare(
        corected_e,
        f_exp=t,
    )[1]

    kl = np.sum(kl_div([item / np.sum(t) for item in t], [item / np.sum(corected_e) for item in corected_e]))

    return chi, kl, t, corected_e, counts