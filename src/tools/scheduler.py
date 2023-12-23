import warnings
import math
from collections import deque
from statistics import mean, median
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


class WarmupLR(StepLR):
    """Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        super(StepLR, self).__init__(optimizer, last_epoch, verbose)

        # rewrite original base learning rate without step()
        for param_groups, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_groups['lr'] = base_lr * 1 / self.warmup_steps

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self._step_count > self.warmup_steps:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return [base_lr * self._step_count / self.warmup_steps for base_lr in self.base_lrs]

    # def _get_closed_form_lr(self):
    #     return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
    #             for base_lr in self.base_lrs]


class ReduceOnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0, coolstart=0,
                 min_lr=0, eps=1e-8, verbose=False):
        super(ReduceOnPlateau, self).__init__(optimizer, mode=mode, factor=factor, patience=patience,
                                              threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown,
                                              min_lr=min_lr, eps=eps, verbose=verbose)

        self._last_metrics = deque([0], maxlen=patience)
        self._last_epoch = 0
        self._last_reduction = 0
        self.best = 0
        self.coolstart = coolstart
        self.stop_early = False

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self._last_epoch + 1
        self._last_epoch = epoch
        self.best = median(self._last_metrics)

        if epoch > self.coolstart:
            if self.is_better(current, self.best):
                # self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                if self.is_better(current, self._last_reduction):
                    self._reduce_lr(epoch)
                    self._last_reduction = self.best
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0
                else:
                    self.stop_early = True

        self._last_metrics.append(metrics)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = 0
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def is_better(self, a, median_score):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < median_score * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < median_score - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > median_score * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > median_score + self.threshold

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))


class ReduceOnPlateauMax(ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0, coolstart=0,
                 min_lr=0, eps=1e-8, verbose=False):
        super(ReduceOnPlateauMax, self).__init__(optimizer, mode=mode, factor=factor, patience=patience,
                                              threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown,
                                              min_lr=min_lr, eps=eps, verbose=verbose)

        self._last_epoch = 0
        self._last_reduction = 0
        self.num_bad_cycles = 0
        self.max_bad_cycles = 3
        self.best = 0
        self.coolstart = coolstart
        self.stop_early = False

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self._last_epoch + 1
        self._last_epoch = epoch

        if epoch < self.coolstart:
            return

        if self.is_better(current, self.best):
            # self.best = current
            self.num_bad_epochs = 0
            self.best = current
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            # check if the best score in the last cycle was better then in previous cycle
            if not self.is_better(self.best, self._last_reduction):
                self.num_bad_cycles += 1

            # if no improvement has been achieved within last `max_bad_cycles` perform early stopping
            if self.num_bad_cycles == self.max_bad_cycles:
                self.stop_early = True
                return

            self._reduce_lr(epoch)
            # restart counters
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

            # store last best score
            self._last_reduction = self.best


        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = 0
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def is_better(self, a, median_score):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < median_score * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < median_score - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > median_score * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > median_score + self.threshold

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))


class PlateauStopper:

    def __init__(
            self,
            patience: int = 10,
            coolstart: int = 10,
            improvement_factor: float = 0.01,
            mode: str = 'max',
            check_finite: bool = True,
            divergence_threshold: float = 0.05,
            divergence_patience: int = 10,
    ):
        super().__init__()
        assert mode in {'min', 'max'}

        self.patience = patience
        self.coolstart = coolstart
        self.mode = mode
        self.stop_early = False
        self.last_eval_score = deque([0], maxlen=patience)
        self.last_train_score = deque([0], maxlen=patience)
        self.check_finite = check_finite
        self.divergence_threshold = divergence_threshold
        self.counter = 0
        self.epoch_counter = 0

        self.message = None

    def reset(self):
        self.counter, self.stop_early = 0, False

    def train_step(self, train_score):
        self.last_train_score.append(train_score)

    def step(self, eval_score):
        self.epoch_counter += 1

        # stop if metric is NaN
        if self.check_finite and math.isnan(eval_score):
            self.stop_early = True
            self.message = 'Stopped by PlateauStopper. Metric is NaN.'
            return

        # ---checking metric divergence---
        if self.epoch_counter > self.coolstart:
            delta = median(self.last_train_score) - self.last_eval_score[-1]
            # stop if train vs eval metrics diverge
            if delta >= self.divergence_threshold:
                self.counter += 1
            else:
                self.counter = 0

        if self.counter >= self.patience:
            self.stop_early = True
            self.message = f'Stopped by PlateauStopper. Metric diverge by {delta} over {self.counter} iterations.'
            return

        self.last_eval_score.append(eval_score)

        
if __name__ == '__main__':
    pass