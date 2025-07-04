import optuna
import torch
import torch.nn as nn
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


#Set seed
SEED = 3052001

ANNOTATION_DIR = "/media/guest/DataStorage/WaveMap/HDF5/annotations_complete.csv"
DATA_DIR = "/media/guest/DataStorage/WaveMap/HDF5"

#hyperparameters
KERNEL_SIZE = (1, 5)
STEM_KERNEL_SIZE = (1, 17)
BLOCKS = [3, 4, 6, 3]
FEATURES = [16, 32, 64, 128]
ACTIVATION = "LReLU"
DOWNSAMPLING_FACTOR = 2
NORMALIZATION = "BatchN2D"
FILTER_UTILIZED = True
SEGMENT_MS = 100
OVERSAMPLING_FACTOR = 4


def sample_cases_controls(risks, events, durations, n_controls):
    risks.squeeze_()
    g_cases = risks[events]
    n_cases = g_cases.numel()
    if n_cases == 1:
        g_cases = g_cases.reshape(1)  # Ensure g_cases is a 1D tensor
    g_controls = torch.zeros(n_controls, n_cases)
    for i in range(n_cases):
        case_time = durations[events][i]
        all_control_risks = risks[durations >= case_time]
        g_control = all_control_risks[torch.randint(0, len(all_control_risks), (n_controls,))]
        g_controls[:, i] = g_control

    return g_cases, g_controls


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def cross_val(folds=3):
    """Cross-validation function to evaluate the model."""

    set_seed(SEED)

    # Hyperparameters to optimize
    projection_nodes = 128
    attention_nodes = 64
    dropout = 0.2
    cox_regularization = 0.01
    learning_rate = 0.001
    weight_decay = 0.01
    batch_size =  32
    num_epochs =  180 #91
    n_controls = 8

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
        "hidden_size": projection_nodes,
        "attention_hidden_size": attention_nodes,
        "output_size": 1,
        "dropout": True,
        "dropout_prob": dropout,
    }

    for fold in range(folds):

        print(f"Fold {fold+1}/{folds}")

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
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_padding)

        # Create a directory for TensorBoard logs
        log_dir = f"runs/cross_val_ns_test_3_/fold_{fold}"
        writer = SummaryWriter(log_dir)

        # Model, Loss, Optimizer
        model = CoxAttentionResnet(resnet_params, amil_params)
        loss_fn = CoxCCLoss(shrink=cox_regularization)
        cindex = ConcordanceIndex()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_epochs // 10,
            num_training_steps=num_epochs, # total number of steps (warmup + cosine decay)
        )
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            total_train_loss = 0.0
            epoch_risks, epoch_events, epoch_durations = list(), list(), list()
            model.train()
            for traces, masks, durations, events in train_dataloader:

                if not events.any():
                    continue

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
            print(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, Training loss: {avg_train_loss:.4f}")
            writer.add_scalar("Train loss", avg_train_loss, epoch)

            #risks_tensor = torch.cat(epoch_risks, dim=0)
            #writer.add_histogram("Train/Risks", risks_tensor, epoch)

            #writer.add_histogram("Train/Attentions", attentions[0,:,:], epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Learning Rate", current_lr, epoch)

            scheduler.step()

            # Training C-index
            epoch_risks = torch.cat(epoch_risks, dim=0)
            epoch_events = torch.cat(epoch_events, dim=0)
            epoch_durations = torch.cat(epoch_durations, dim=0)
            concordance_train = cindex(estimate=epoch_risks, event=epoch_events, time=epoch_durations)
            print(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, Training C-index: {concordance_train:.4f}")
            writer.add_scalar("C-index training", concordance_train, epoch)

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_epoch_risks, val_epoch_events, val_epoch_durations = list(), list(), list()
                for val_traces, val_masks, val_durations, val_events in val_dataloader:

                    if not val_events.any():
                        continue

                    val_risks, val_attentions = model(val_traces, val_masks)

                    val_g_cases, val_g_controls = sample_cases_controls(val_risks, val_events, val_durations, n_controls)

                    val_loss += loss_fn(val_g_cases, val_g_controls).item()

                    val_epoch_risks.append(val_risks.detach().cpu().view(-1))
                    val_epoch_events.append(val_events.detach().cpu().view(-1))
                    val_epoch_durations.append(val_durations.detach().cpu().view(-1))

                avg_val_loss = val_loss / len(val_dataloader)
                print(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, Validation loss: {avg_val_loss:.4f}")
                writer.add_scalar("Validation loss", avg_val_loss, epoch)

            # Validation C-index
            val_epoch_risks = torch.cat(val_epoch_risks, dim=0)
            val_epoch_events = torch.cat(val_epoch_events, dim=0)
            val_epoch_durations = torch.cat(val_epoch_durations, dim=0)
            concordance_val = cindex(estimate=val_epoch_risks, event=val_epoch_events, time=val_epoch_durations)
            print(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, Validation C-index: {concordance_val:.4f}")
            writer.add_scalar("C-index evaluation", concordance_val, epoch)

        # Log best values (e.g., final validation cindex and training loss)
        writer.add_hparams(
            {"projection_nodes": projection_nodes,
            "attention_nodes": attention_nodes,
            "dropout": dropout,
            "cox_regularization": cox_regularization,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs},
            {"hparam/concordance_val": concordance_val, 
            "hparam/loss": avg_train_loss,
            "hparam/val_loss": avg_val_loss,
            "hparam/train_cindex": concordance_train}
        )

        writer.close()

        torch.save(model.state_dict(), f"/home/guest/lib/data/saved_models/cross_val_new_sampling_3_{fold}.pth")


if __name__ == "__main__":
    # Cross-validation
    folds = 3
    cross_val(folds=folds)
