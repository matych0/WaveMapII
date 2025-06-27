import optuna
import os
import torch
import torch.optim as optim
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataset.dataset import EGMDataset
from src.dataset.collate import collate_padding
from model.cox_mil_resnet import CoxAttentionResnet
from losses.loss import CoxCCLoss
from src.transforms.transforms import (BaseTransform, RandomAmplifier, RandomGaussian,
                                       RandomTemporalScale, RandomShift, TanhNormalize,
                                       RandomPolarity)
import numpy as np
from torchsurv.metrics.cindex import ConcordanceIndex
from statistics import median


#Set seed
SEED = 3052001

ANNOTATION_DIR = "D:/Matych/HDF5/annotations_complete.csv"
DATA_DIR = "D:/Matych/HDF5"

SAVE_MODEL_PATH = "D:/Matych/saved_models"
STUDY_NAME = "hyperparam_optim_mult_controls_CUDA"
TB_LOG_DIR = "C:/Users/xmatyc00/Diplomka/runs"
STUDY_DIR = "C:/Users/xmatyc00/Diplomka/optuna_studies"



#hyperparameters
KERNEL_SIZE = (1, 5)
STEM_KERNEL_SIZE = (1, 17)
BLOCKS = [3, 4, 6, 3]
FEATURES = [16, 32, 64, 128]
PROJECTION_NODES = 128
ATTENTION_NODES = 64
ACTIVATION = "LReLU"
DOWNSAMPLING_FACTOR = 2
NORMALIZATION = "BatchN2D"
FILTER_UTILIZED = True
SEGMENT_MS = 100
OVERSAMPLING_FACTOR = None
CHUNK_SIZE = 8
FOLDS = 3



def sample_cases_controls(risks, events, durations, n_controls):
    device = risks.device

    risks.squeeze_()
    g_cases = risks[events]
    n_cases = g_cases.numel()
    if n_cases == 1:
        g_cases = g_cases.reshape(1)  # Ensure g_cases is a 1D tensor
    g_controls = torch.zeros(n_controls, n_cases, device=device)
    for i in range(n_cases):
        case_time = durations[events][i]
        all_control_risks = risks[durations >= case_time]
        g_control = all_control_risks[torch.randint(0, len(all_control_risks), (n_controls,), device=device)]
        g_controls[:, i] = g_control

    return g_cases, g_controls


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters to optimize
    dropout = trial.suggest_categorical('dropout', [0.0, 0.25, 0.5])
    cox_regularization = trial.suggest_categorical('cox_regularization', [0.0, 1e-3, 1e-2, 1e-1])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-4, 1e-3, 1e-2, 1e-1])
    batch_size =  trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 50, 500, log=True)
    n_controls = trial.suggest_categorical("n_controls", [4, 8])

    # Transformations
    polarity = RandomPolarity(probability=0.5, shuffle=True, random_seed=SEED)
    temporal_scale = RandomTemporalScale(probability=0.2, limit=0.2, shuffle=True, random_seed=SEED)
    amplifier = RandomAmplifier(probability=0.2, limit=0.2, shuffle=True, random_seed=SEED)
    noise = RandomGaussian(probability=0.2, low_limit=10, high_limit=30, shuffle=True, random_seed=SEED)
    shift = RandomShift(probability=0.5, shift_range=0.3, shuffle=True, random_seed=SEED)
    tanh_normalize = TanhNormalize(factor=5)
    shuffle = BaseTransform(shuffle=True, random_seed=SEED)

    # Define parameters for LocalActivationResNet
    resnet_params = {
        "in_features": 1,
        "kernel_size": KERNEL_SIZE,
        "stem_kernel_size": STEM_KERNEL_SIZE,
        "blocks": BLOCKS,
        "features": FEATURES,
        "activation": ACTIVATION,
        "downsampling_factor": DOWNSAMPLING_FACTOR,
        "normalization": NORMALIZATION,
        "preactivation": False,
        "trace_stages": True,
    }

    # Define parameters for AMIL
    amil_params = {
        "input_size": FEATURES[-1],
        "hidden_size": PROJECTION_NODES,
        "attention_hidden_size": ATTENTION_NODES,
        "output_size": 1,
        "dropout": True,
        "dropout_prob": dropout,
    }


    print(f"Optuna Trial {trial.number}")
    print(f"Epochs: {num_epochs}")

    fold_concordances = []

    for fold in range(FOLDS):

        train_transform = transforms.Compose([
            polarity,
            amplifier,
            temporal_scale,
            shift,
            noise,
            shuffle,
            tanh_normalize,
        ])

        val_transform = transforms.Compose([
            tanh_normalize,
        ])

        # Dataset & Dataloader
        train_dataset = EGMDataset(
            annotations_file=ANNOTATION_DIR,
            data_dir=DATA_DIR,
            startswith="LA",
            training=True,
            transform=train_transform,
            readjustonce=True,
            segment_ms=SEGMENT_MS,
            filter_utilized=FILTER_UTILIZED,
            #oversampling_factor=OVERSAMPLING_FACTOR,
            controls_time_shift=60,  # Shift controls by 60 days
            fold=fold,
            random_seed=SEED,
        )

        val_dataset = EGMDataset(
            annotations_file=ANNOTATION_DIR,
            data_dir=DATA_DIR,
            startswith="LA",
            training=False,
            transform=val_transform,
            readjustonce=True,
            segment_ms=SEGMENT_MS,
            filter_utilized=FILTER_UTILIZED,
            #oversampling_factor=OVERSAMPLING_FACTOR,
            controls_time_shift=60,  # Shift controls by 60 days
            fold=fold,
            random_seed=SEED,
        )

        # Create DataLoader
        generator = torch.Generator()
        generator.manual_seed(SEED)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=collate_padding)
        val_batch_size = batch_size if batch_size <= CHUNK_SIZE else CHUNK_SIZE
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_padding)

        # Create a directory for TensorBoard logs
        log_dir = os.path.join(TB_LOG_DIR, STUDY_NAME, f"optuna_trial_{trial.number}/fold_{fold}")
        writer = SummaryWriter(log_dir)

        # Model, Loss, Optimizer
        model = CoxAttentionResnet(resnet_params, amil_params).to(device)
        loss_fn = CoxCCLoss(shrink=cox_regularization).to(device)
        cindex = ConcordanceIndex()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_epochs // 10,
            num_training_steps=num_epochs, # total number of steps (warmup + cosine decay)
        )

        val_concordances = []
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            total_train_loss = 0.0
            epoch_risks, epoch_events, epoch_durations = list(), list(), list()
            model.train()
            for traces, masks, durations, events in train_dataloader:
                traces = traces.to(device)
                masks = masks.to(device)
                durations = durations.to(device)
                events = events.to(device)

                if not events.any():
                    continue

                # Gradient accumulation
                if batch_size > CHUNK_SIZE:

                    accumulation_steps = batch_size // CHUNK_SIZE
                    optimizer.zero_grad()
                    batch_loss = 0.0  # To accumulate loss for logging

                    for i in range(accumulation_steps):
                        start = i * CHUNK_SIZE
                        end = min(start + CHUNK_SIZE, batch_size)

                        events_chunk = events[start:end]

                        if not events_chunk.any():
                            continue

                        traces_chunk = traces[start:end]
                        masks_chunk = masks[start:end]
                        durations_chunk = durations[start:end]

                        risks_chunk, attentions_chunk = model(traces_chunk, masks_chunk)

                        g_cases, g_controls = sample_cases_controls(risks_chunk, events_chunk, durations_chunk,
                                                                    n_controls)
                        loss = loss_fn(g_cases, g_controls)

                        # Normalize loss to maintain consistent scale
                        loss = loss / accumulation_steps
                        loss.backward()

                        batch_loss += loss.item()

                        # Store for logging
                        epoch_risks.append(risks_chunk.detach().cpu().view(-1))
                        epoch_events.append(events_chunk.detach().cpu().view(-1))
                        epoch_durations.append(durations_chunk.detach().cpu().view(-1))

                    optimizer.step()
                    total_train_loss += batch_loss

                else:
                    risks, attentions = model(traces, masks)

                    g_cases, g_controls = sample_cases_controls(risks, events, durations, n_controls)

                    loss = loss_fn(g_cases, g_controls)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()

                    epoch_risks.append(risks.detach().cpu().view(-1))
                    epoch_events.append(events.detach().cpu().view(-1))
                    epoch_durations.append(durations.detach().cpu().view(-1))

            avg_train_loss = total_train_loss / len(train_dataloader)
            writer.add_scalar("Train loss", avg_train_loss, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Learning Rate", current_lr, epoch)

            scheduler.step()

            # Training C-index
            epoch_risks = torch.cat(epoch_risks, dim=0)
            epoch_events = torch.cat(epoch_events, dim=0)
            epoch_durations = torch.cat(epoch_durations, dim=0)
            concordance_train = cindex(estimate=epoch_risks, event=epoch_events, time=epoch_durations)
            writer.add_scalar("C-index training", concordance_train, epoch)

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_epoch_risks, val_epoch_events, val_epoch_durations = list(), list(), list()
                for val_traces, val_masks, val_durations, val_events in val_dataloader:
                    
                    val_traces = val_traces.to(device)
                    val_masks = val_masks.to(device)
                    val_durations = val_durations.to(device)
                    val_events = val_events.to(device)

                    if not val_events.any():
                        continue

                    val_risks, val_attentions = model(val_traces, val_masks)

                    val_g_cases, val_g_controls = sample_cases_controls(val_risks, val_events, val_durations, n_controls)

                    val_loss += loss_fn(val_g_cases, val_g_controls).item()

                    val_epoch_risks.append(val_risks.detach().cpu().view(-1))
                    val_epoch_events.append(val_events.detach().cpu().view(-1))
                    val_epoch_durations.append(val_durations.detach().cpu().view(-1))

                avg_val_loss = val_loss / len(val_dataloader)
                writer.add_scalar("Validation loss", avg_val_loss, epoch)

            # Validation C-index
            val_epoch_risks = torch.cat(val_epoch_risks, dim=0)
            val_epoch_events = torch.cat(val_epoch_events, dim=0)
            val_epoch_durations = torch.cat(val_epoch_durations, dim=0)
            concordance_val = cindex(estimate=val_epoch_risks, event=val_epoch_events, time=val_epoch_durations)
            val_concordances.append(concordance_val)
            writer.add_scalar("C-index evaluation", concordance_val, epoch)

        fold_concordance = median(val_concordances[-10:])  # Use median of last 10 epochs
        print(f"Fold {fold}, Validation C-index: {fold_concordance:.4f}")
        print(f"Fold {fold}, Training C-index: {concordance_train:.4f}")
        fold_concordances.append(fold_concordance)

        # Log best values (e.g., final validation cindex and training loss)
        writer.add_hparams(
            {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_controls": n_controls,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "cox_regularization": cox_regularization,
            "num_epochs": num_epochs
            },
            {"hparam/val_cindex": fold_concordance, 
            "hparam/loss": avg_train_loss,
            "hparam/val_loss": avg_val_loss,
            "hparam/train_cindex": concordance_train}
        )

        writer.close()

        if fold_concordance > 0.55:  #####!
            torch.save(model.state_dict(), os.path.join(SAVE_MODEL_PATH, f"{STUDY_NAME}_trial_{trial.number}_fold_{fold}.pth"))

        # Pruning after poor folds
        trial.report(fold_concordance, step=fold)
        if trial.should_prune() and fold < 2:
            raise optuna.TrialPruned()

    mean_concordance = np.mean(fold_concordances)
    print(f"Overall mean C-index: {mean_concordance:.4f}")

    return mean_concordance  # Maximize concordance → minimize negative


if __name__ == "__main__":

    num_trials = 200

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=3052001, n_startup_trials=20)

    pruner = optuna.pruners.ThresholdPruner(
        lower=0.5,  # Prune if fold C-index value is < 0.5
        upper=None
    )

    study_path = os.path.join(STUDY_DIR, STUDY_NAME).replace("\\", "/")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage="sqlite:///" + study_path,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    study.optimize(objective, n_trials=num_trials)
    print("Best hyperparameters:", study.best_params)