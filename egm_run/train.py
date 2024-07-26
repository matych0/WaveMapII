import os
import logging
import json

from pathlib import Path
from collections import defaultdict

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
import numpy as np


from src.dataset import generator
from src.transforms import transforms as tfs

from src.transforms.encodings import (
    to_instances,
    to_onehot,
    resample,
)

from src.tools.scheduler import (
    WarmupLR,
    PlateauStopper,
)
from src.tools.tools import BestModel

from src.tools.metrics import (
    AvgLoss,
    ConfusionMatrix,
    BinaryMetrics,
    RegressionMetrics,
    StatsWarehouse,
    PerFileWarehouse,
)

from src.tools.tools import output_to_sel

#path to a folder, where the current script is stored 
base_path = Path(__file__).parent


class StatsLogger:
    """
    Custom logger for storing epoch-based model statistics
    """
    def __init__(self):
        self.metrics = defaultdict(list)

    def update(self, item_name, item):
        self.metrics[item_name].append(item)


# Instantiate evaluation metrics
avgloss_ = AvgLoss()
cfm_ = ConfusionMatrix(nb_classes=1, normalize=False, limit=0.5)
bin_metrics = BinaryMetrics(cfm=cfm_, metrics=('f_score'))
onset_metrics = RegressionMetrics(limit=35, unit_scale=0.5)
offset_metrics = RegressionMetrics(limit=35, unit_scale=0.5)
perfile_cfm = PerFileWarehouse(limit=0.5)

warehouse = StatsWarehouse(
    cols=(
        'Epoch', 'Loop', 'LRate', 'Loss', 'F1',
        'TP', 'FP', 'TN', 'FN',
        'Rec_On', 'P+On', 'RMSE_On', 'Rec_Off', 'P+Off', 'RMSE_Off',
        ),
    formatting=(
        '03d', None, '.8f', '.8f', '.4f',
        '04d', '04d', '04d', '04d',
        '.4f', '.4f', '.2f', '.4f', '.4f', '.2f',
        ),
    print_cols=(
        'Epoch', 'Loop', 'LRate', 'Loss', 'F1',
        'Rec_On', 'P+On', 'RMSE_On', 'Rec_Off', 'P+Off', 'RMSE_Off',
        ),
    )


class LoopWrapper:
    def __init__(
            self,
            model,
            device,
            optimizer,
            scheduler,
            **kwargs,
            ):
        
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler        
        self.criterion = kwargs.pop('criterion', None)        
        self.cfg = kwargs.pop('cfg', None)        
        self.epoch = None

    def step(self, x, targets, masks, **kwargs):
        batch_size, _, max_len = x.shape

        self.epoch = kwargs.pop('epoch', None)
        widths = kwargs.pop('widths', [])
        batch_files = kwargs.pop('batch_files', [])
        labels = kwargs.pop('labels', None)
        cos_w = kwargs.pop('w', None)

        x = x.to(self.device)

        # One Hot encoding
        resampling_factor = 1
        resampled_targets = resample(targets, factor=resampling_factor)

        # torch.autograd.set_detect_anomaly(True)

        # Forward pass
        y = self.model(x)
        y0 = torch.sigmoid(y)

        # One Hot encoding
        t0 = to_onehot(resampled_targets, y.shape[-1])
        t0 = torch.FloatTensor(t0).to(x.device)

        # create masks
        masks = torch.zeros(batch_size, y.shape[-1], dtype=torch.bool).to(self.device)
        for b, w in enumerate(widths):
            masks[b, :int(resampling_factor * w)] = True

        if self.cfg.advanced_features.gradient_mask and self.model.training:
            y.register_hook(lambda grad: grad * masks.float())

        loss = self.criterion['tversky'](y0, t0, masks)

        # Update model
        if self.model.training:
            self.optimizer.zero_grad()
            loss.backward()

            if self.cfg.advanced_features.gradient_clipping is not None:
                clip_grad_norm_(self.model.parameters(), 5)

            self.optimizer.step()
            self.scheduler.step()

        loss = loss.detach().cpu().numpy()
        y0 = y0.detach().cpu().numpy()
        t0 = t0.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()

        if not self.model.training:
            pass

        # Update eval metrics for training subset
        avgloss_.update(loss, batch_size)
        cfm_.update(y0, t0, masks)

        # convert one hot encoding to instances
        instance_estimates = to_instances(y0, threshold=0.5)

        # store
        for unbatched_y, unbatched_t0, unbatched_masks, unbatched_estimates, unbatched_targets, unbatched_labels, file_name in zip(
            y0,
            t0,
            masks,
            instance_estimates,
            targets,
            labels,
            batch_files,
            ):            

            oid = file_name.lstrip(self.cfg.dataset.path[0]).rstrip('.h5')

            if self.model.training:
                # Does not store labels in the training loop due to RandomShift possibly resulting in length mismatch
                groups = ['NA'] * len(unbatched_targets)
            else:
                groups = unbatched_labels[-1][-1]

            if not unbatched_estimates:
                unbatched_estimates = np.array([[], []])
            else:
                unbatched_estimates = np.array(unbatched_estimates).transpose()

            if not unbatched_targets:
                unbatched_targets = np.array([[], []])
            else:
                unbatched_targets = np.array(unbatched_targets).transpose()

            if not groups:
                groups = ['AF']

            perfile_cfm.update(
                unbatched_y,
                unbatched_t0,
                unbatched_masks,
                groups='WAF' if 'AF' in groups else 'WOAF',
                oid=oid,
            )
            try:
                onset_metrics.update(
                    y=unbatched_estimates[0, :],
                    targets=unbatched_targets[0, :],
                    groups=groups,
                    oid=oid,
                )

                offset_metrics.update(
                    y=unbatched_estimates[1, :],
                    targets=unbatched_targets[1, :],
                    groups=groups,
                    oid=oid,
                )
            except IndexError:
                print()

    def compute_metrics(self):
        binary_stats = bin_metrics.compute()

        warehouse.update(
            [
                self.epoch,
                'TRAIN' if self.model.training else 'VALID',
                self.optimizer.param_groups[0]["lr"],
                avgloss_.compute(),
                binary_stats['f_score'],
                int(cfm_._cfm[0][0]),
                int(cfm_._cfm[0][1]),
                int(cfm_._cfm[0][2]),
                int(cfm_._cfm[0][3]),
                onset_metrics.recall(),
                onset_metrics.precision(),
                onset_metrics.rmse(),
                offset_metrics.recall(),
                offset_metrics.precision(),
                offset_metrics.rmse(),
            ]
        )

    def reset_metrics(self):
        avgloss_.reset()
        cfm_.reset()
        onset_metrics.reset()
        offset_metrics.reset()
        perfile_cfm.reset()


def train_eval_fcn(mdl, dataset, cfg, model_name: str = 'mdl') -> tuple:

    # CUDA for PyTorch
    device = torch.device('cuda:0' if cfg.device.use_cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    # cudnn.benchmark = cfg.device.benchmark

    storage_path = os.path.join(HydraConfig.get().run.dir, str(HydraConfig.get().job.num), cfg.dataset.subset_name)

    # Preprocessing transforms
    if 'transforms' in cfg:
        # with augmentation for training loop
        logger.info('Creating data preprocessing pipeline for training loop.')
        pipeline_train = tfs.Compose([
            instantiate(item) for _, item in cfg.transforms.items()
        ])
        # without augmentation for evaluation loop
        logger.info('Creating data preprocessing pipeline for evaluation loop.')
        pipeline_eval = tfs.Compose([
            instantiate(item) for _, item in cfg.transforms.items() if not item.augmentation
        ])
    else:
        logger.info('No preprocessing steps will be applied to samples.')
        pipeline_train, pipeline_eval = None, None

    # Create training set
    training_set = generator.Dataset(
        dataset['train'],
        channels=cfg.dataset.channels,
        marks=cfg.dataset.mark_groups,
        transform=pipeline_train,
    )

    training_generator = data.DataLoader(training_set, **cfg.batch_params)
    # Create Collate object for training generator
    training_generator.collate_fn = instantiate(training_generator.collate_fn)

    logger.info('Creating training set generator...ok')

    # Create evaluation set
    evaluation_set = generator.Dataset(
        dataset['evaluation'],
        channels=cfg.dataset.channels,
        marks=cfg.dataset.mark_groups,
        transform=pipeline_eval,
    )
    eval_generator = data.DataLoader(evaluation_set, **cfg.batch_params)
    # Create Collate object for eval generator
    eval_generator.collate_fn = instantiate(eval_generator.collate_fn)

    # Create test set 1
    if 'test' in dataset:
        test_set = generator.Dataset(
            dataset['test'],
            channels=cfg.dataset.channels,
            marks=cfg.dataset.mark_groups,
            transform=pipeline_eval,
        )
        test_generator = data.DataLoader(test_set, **cfg.batch_params)
        # Create Collate object for eval generator
        test_generator.collate_fn = instantiate(test_generator.collate_fn)

    logger.info('Creating evaluation set generator...ok')

    # Send model to device
    mdl = mdl.to(device)

    # Instantiate optimizer
    optimizer = instantiate(cfg.optimizer, mdl.parameters())

    # Instantiate learning rate scheduler
    lr_scheduler = instantiate(cfg.scheduler, optimizer)

    # Warmup steps from ADAMs beta2
    warmup_steps = cfg.train_params.warmup_epochs * len(training_generator.dataset) // training_generator.batch_size
    warmup_scheduler = WarmupLR(optimizer, warmup_steps=warmup_steps)

    # Instantiate early stopping
    early_stopper = PlateauStopper(**cfg.early_stopping)

    best_model = BestModel(mdl)

    # Loop over epochs
    logger.info(f'Beginning the training of {cfg.model._target_} model.')

    criterion = {
        'tversky': instantiate(cfg.loss).to(device),
    }

    model_loop = LoopWrapper(
        model=mdl,
        device=device,
        optimizer=optimizer,
        scheduler=warmup_scheduler,
        criterion=criterion,        
        cfg=cfg,
    )

    # Create set of evaluation epoch indices
    eval_epochs = set(range(
        cfg.train_params.warmup_epochs + 1,
        cfg.train_params.epochs,
        cfg.train_params.eval_interval,
        ))
    
    last_epoch = False
    for epoch in range(cfg.train_params.epochs):
        # Indicate last epoch
        if epoch == cfg.train_params.epochs - 1:
            last_epoch = True

        model_loop.model.train()

        # ------------------------ TRAIN -----------------------
        for x, targets, widths, masks, labels, batch_files in training_generator:
            model_loop.step(
                x,
                targets,
                masks,
                epoch=epoch,
                widths=widths,
                batch_files=batch_files,
                labels=labels,
                )

        # compute training statistics
        model_loop.compute_metrics()
        model_loop.reset_metrics()        
        
        logger.info(warehouse.stringify(-1))

        # early_stopper.train_step(scores[cfg.train_params.tracked_metric])
        last_score = warehouse.df[cfg.trial.tracked_metric].iloc[-1]
        early_stopper.train_step(last_score)

        model_loop.reset_metrics()

        # ------------------------ EVAL -----------------------
        if epoch in eval_epochs or last_epoch:

            model_loop.model.eval()
            with torch.set_grad_enabled(False):
                for x, targets, widths, masks, labels, batch_files in eval_generator:
                    model_loop.step(
                        x,
                        targets,
                        masks,
                        epoch=epoch,
                        widths=widths,
                        batch_files=batch_files,
                        labels=labels,
                        )

                    # store results of inference on evaluation set every n-th epoch
                    if cfg.output_to_sel and epoch % 10 == 0:
                        output_to_sel(
                            model_loop.tracked_output,
                            batch_files,
                        )

            # compute training statistics
            model_loop.compute_metrics()

            logger.info(warehouse.stringify(-1))

            # update process handlers
            last_score = warehouse.df[cfg.trial.tracked_metric].iloc[-1]
            best_model.update(last_score)
            lr_scheduler.step(last_score)
            early_stopper.step(last_score)

        if cfg.trial.store_model:
            if best_model.is_updated:
                best_model.store(path=storage_path)
                warehouse.store(path=os.path.join(storage_path, 'warehouse_BEST.csv'))
                onset_metrics.store(path=os.path.join(storage_path, 'onsets_BEST.csv'))
                offset_metrics.store(path=os.path.join(storage_path, 'offsets_BEST.csv'))
                perfile_cfm.store(path=os.path.join(storage_path, 'cfm_BEST.csv'))
                logger.info(f'New best model has been stored in epoch: {epoch}')

            if (
                last_epoch or
                early_stopper.stop_early or
                lr_scheduler.stop_early
            ):
                best_model.store(path=storage_path, suffix='_END')
                warehouse.store(path=os.path.join(storage_path, 'warehouse_END.csv'))
                onset_metrics.store(path=os.path.join(storage_path, 'onsets_END.csv'))
                offset_metrics.store(path=os.path.join(storage_path, 'offsets_END.csv'))
                perfile_cfm.store(path=os.path.join(storage_path, 'cfm_END.csv'))
                logger.info(f'Final model has been stored')

        model_loop.reset_metrics()

        # ------------------------ TEST -----------------------
        if 'test' in dataset:
            if epoch in eval_epochs or last_epoch:

                model_loop.model.eval()
                with torch.set_grad_enabled(False):
                    for x, targets, widths, masks, labels, batch_files in test_generator:
                        model_loop.step(
                            x,
                            targets,
                            masks,
                            epoch=epoch,
                            widths=widths,
                            batch_files=batch_files,
                            labels=labels,
                        )

                # compute training statistics
                model_loop.compute_metrics()

            if cfg.trial.store_model:
                if best_model.is_updated:
                    warehouse.store(path=os.path.join(storage_path, 'warehouse-test_BEST.csv'))
                    onset_metrics.store(path=os.path.join(storage_path, 'onsets-test_BEST.csv'))
                    offset_metrics.store(path=os.path.join(storage_path, 'offsets-test_BEST.csv'))
                    perfile_cfm.store(path=os.path.join(storage_path, 'cfm-test_BEST.csv'))
                if (
                        last_epoch or
                        early_stopper.stop_early or
                        lr_scheduler.stop_early
                ):
                    warehouse.store(path=os.path.join(storage_path, 'warehouse-test_END.csv'))
                    onset_metrics.store(path=os.path.join(storage_path, 'onsets-test_END.csv'))
                    offset_metrics.store(path=os.path.join(storage_path, 'offsets-test_END.csv'))
                    perfile_cfm.store(path=os.path.join(storage_path, 'cfm-test_END.csv'))

            model_loop.reset_metrics()

        # activate early stopping
        if early_stopper.stop_early:
            logger.info(early_stopper.message)
            break

        if lr_scheduler.stop_early:
            logger.info(f'Stopped by PlateauScheduler. Metric did not improve two reduction cycles in a row.')
            break

    return last_score, best_model.best_score


@hydra.main(config_path='config', config_name='config', version_base=None)
def run_experiment(cfg: DictConfig) -> tuple:        
    
    # Check if path to training data exists    
    dataset_path = cfg.dataset.path[0]
    if not os.path.isdir(dataset_path):
        raise IOError(f'Dataset folder name {dataset_path} does not exist.')
    
    # Read dataset partitions
    with open(os.path.join(os.getcwd(), 'config', 'dataset', cfg.dataset.partitions), 'r') as f_obj:
        partitions = json.load(f_obj)

    ds_folds = list()
    for fold_name in ('fold_0', 'fold_1', 'fold_2'):
        partitions[fold_name] = [os.path.join(dataset_path, item) for item in partitions[fold_name]]

    # Create folds
    fold_abbrevs = ('f0', 'f1', 'f2')
    ds_folds = [
        {
            'train': partitions['fold_0'] + partitions['fold_1'],
            'evaluation': partitions['fold_2'],
        },
        {
            'train': partitions['fold_0'] + partitions['fold_2'],
            'evaluation': partitions['fold_1'],
        },
        {
            'train': partitions['fold_1'] + partitions['fold_2'],
            'evaluation': partitions['fold_0'],
        },
    ]

    # perform multiple trials with single model
    trial_scores = []
    for trial_idx in range(cfg.trial.repeat_trial):    

        # iterate over each fold if cross-validation dataset is used
        fold_scores = []
        for i, (ds_fold, fold_name) in enumerate(zip(ds_folds, fold_abbrevs)):

            try:
                os.makedirs(os.path.join(HydraConfig.get().run.dir, str(HydraConfig.get().job.num), fold_name), exist_ok=True)
                cfg.dataset.subset_name = fold_name
            except OSError as e:
                logger.info(e)
                raise

            torch.cuda.empty_cache()

            # Instantiate model before each run
            logger.info(f'Instantiating model...')

            torch.manual_seed(cfg.device.seed)
            logger.info(f'Torch seed: {cfg.device.seed}')
            #Instantiate Pyramid ResNet with parameters stored in YAML file
            mdl = instantiate(cfg.model)

            params = sum(p.numel() for p in mdl.parameters() if p.requires_grad)           

            try:
                # Train end evaluate model
                logger.info(f'Training job: {HydraConfig.get().job.num}; trial: {trial_idx+1}; subfold {i}.')
                end_score, best_score = train_eval_fcn(mdl, ds_fold, cfg, trial_idx)

            except RuntimeError as e:
                logger.info(f'Error during training: {e}.')
                
            warehouse.reset()

            fold_scores.append(end_score)
        trial_scores.append(fold_scores)
        # --- end of repeated trial with same hyperparameters ---

    # average multiple runs
    if np.all(np.isnan(trial_scores)):
        out_score = np.NINF
    else:
        out_score = np.nanmedian(trial_scores)

    return out_score


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    stats_log = StatsLogger()

    run_experiment()


