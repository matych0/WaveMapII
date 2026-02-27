import torch
import hydra
from torchsurv.metrics.cindex import ConcordanceIndex


def sample_cases_controls(risks, events, durations, n_controls):
    """Sampling identical to avg_pooling.py and best_trial.py."""
    device = risks.device

    risks = risks.view(-1)
    events = events.view(-1)
    durations = durations.view(-1)

    g_cases = risks[events]
    if g_cases.numel() == 1:
        g_cases = g_cases.reshape(1)

    n_cases = g_cases.numel()
    g_controls = torch.zeros(n_controls, n_cases, device=device)

    for i in range(n_cases):
        case_time = durations[events][i]
        valid_controls = risks[durations >= case_time]
        idx = torch.randint(0, len(valid_controls), (n_controls,), device=device)
        g_controls[:, i] = valid_controls[idx]

    return g_cases, g_controls


class SurvivalTask:

    def __init__(self, cfg, device):
        self.loss_fn = hydra.utils.instantiate(cfg.training.loss)
        self.cindex = ConcordanceIndex()
        self.n_controls = cfg.training.hparams.n_controls

        self.reset()

    def reset(self):
        self.train_risks = []
        self.train_events = []
        self.train_durations = []
        self.val_risks = []
        self.val_events = []
        self.val_durations = []

    def compute_loss(self, risks, durations, events):
        if not events.any():
            return None
        g_cases, g_controls = sample_cases_controls(risks, events, durations, self.n_controls)
        return self.loss_fn(g_cases, g_controls)

    def update_train_metrics(self, risks, durations, events):
        self.train_risks.append(risks.detach().cpu().view(-1))
        self.train_events.append(events.cpu().view(-1))
        self.train_durations.append(durations.cpu().view(-1))

    def update_val_metrics(self, risks, durations, events):
        self.val_risks.append(risks.detach().cpu().view(-1))
        self.val_events.append(events.cpu().view(-1))
        self.val_durations.append(durations.cpu().view(-1))

    def compute_epoch_train_metrics(self):
        r = torch.cat(self.train_risks)
        e = torch.cat(self.train_events)
        t = torch.cat(self.train_durations)
        return {"cindex": self.cindex(r, e, t)}

    def compute_epoch_val_metrics(self):
        r = torch.cat(self.val_risks)
        e = torch.cat(self.val_events)
        t = torch.cat(self.val_durations)
        return {"cindex": self.cindex(r, e, t)}
