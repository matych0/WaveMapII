import torch
import hydra
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

class ClassificationTask:

    def __init__(self, cfg, device):
        self.loss_fn = hydra.utils.instantiate(cfg.training.loss).to(device)
        self.reset()

    def reset(self):
        self.train_logits = []
        self.train_targets = []
        self.val_logits = []
        self.val_targets = []

    def compute_loss(self, logits, durations, events):
        targets = events.float()
        return self.loss_fn(logits.view(-1), targets.view(-1))

    def update_train_metrics(self, logits, durations, events):
        self.train_logits.append(logits.detach().cpu().view(-1))
        self.train_targets.append(events.cpu().view(-1))

    def update_val_metrics(self, logits, durations, events):
        self.val_logits.append(logits.detach().cpu().view(-1))
        self.val_targets.append(events.cpu().view(-1))

    def compute_epoch_train_metrics(self):
        logits = torch.cat(self.train_logits)
        y = torch.cat(self.train_targets)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).numpy()
        return {
            "auc": roc_auc_score(y.numpy(), probs.numpy()),
            "acc": accuracy_score(y.numpy(), pred),
            "f1": f1_score(y.numpy(), pred),
        }

    def compute_epoch_val_metrics(self):
        logits = torch.cat(self.val_logits)
        y = torch.cat(self.val_targets)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).numpy()
        return {
            "auc": roc_auc_score(y.numpy(), probs.numpy()),
            "acc": accuracy_score(y.numpy(), pred),
            "f1": f1_score(y.numpy(), pred),
        }
