import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsurv.metrics.cindex import ConcordanceIndex
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup

from losses.loss import CoxCCLoss
from model.cox_mil_resnet import CoxAttentionResnet
from src.dataset.collate import collate_padding
from src.dataset.dataset import EGMDataset
from src.transforms.transforms import TanhNormalize

# Set seed
SEED = 3052001

ANNOTATION_DIR = ""
DATA_DIR = ""

SAVE_MODEL_PATH = ""
STUDY_NAME = "min_max_normalization"
TB_LOG_DIR = ""

# hyperparameters
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

# Optimized hyperparameters
DROPOUT = 0.5
COX_REGULARIZATION = 0.01
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 32
NUM_EPOCHS = 264
N_CONTROLS = 4


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


def cross_val(folds=3):
    """Cross-validation function to evaluate the model."""

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accumulation_steps = BATCH_SIZE // CHUNK_SIZE

    # Transformations
    tanh_normalize = TanhNormalize(factor=5)

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

    # Define parameters for MAXMIL
    amil_params = {
        "input_size": FEATURES[-1],
        "hidden_size": PROJECTION_NODES,
        "attention_hidden_size": ATTENTION_NODES,
        "output_size": 1,
        "dropout": True,
        "dropout_prob": DROPOUT,
    }

    for fold in range(folds):

        print(f"Fold {fold + 1}/{folds}")

        train_transform = val_transform = transforms.Compose([
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
            fold=fold,
            random_seed=SEED,
        )

        # Create DataLoader
        generator = torch.Generator()
        generator.manual_seed(SEED)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
                                      collate_fn=collate_padding)

        val_batch_size = BATCH_SIZE if BATCH_SIZE <= CHUNK_SIZE else CHUNK_SIZE
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_padding)

        # Create a directory for TensorBoard logs
        log_dir = os.path.join(TB_LOG_DIR, STUDY_NAME, f"fold_{fold}")
        writer = SummaryWriter(log_dir)

        # Model, Loss, Optimizer
        model = CoxAttentionResnet(resnet_params, amil_params).to(device)
        loss_fn = CoxCCLoss(shrink=COX_REGULARIZATION).to(device)
        cindex = ConcordanceIndex()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=NUM_EPOCHS // 10,
            num_training_steps=NUM_EPOCHS,
        )

        # Training loop
        model.train()
        for epoch in range(NUM_EPOCHS):
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
                if BATCH_SIZE > CHUNK_SIZE:

                    accumulation_steps = BATCH_SIZE // CHUNK_SIZE
                    optimizer.zero_grad()
                    batch_loss = 0.0  # To accumulate loss for logging

                    for i in range(accumulation_steps):
                        start = i * CHUNK_SIZE
                        end = min(start + CHUNK_SIZE, BATCH_SIZE)

                        events_chunk = events[start:end]

                        if not events_chunk.any():
                            continue

                        traces_chunk = traces[start:end]
                        masks_chunk = masks[start:end]
                        durations_chunk = durations[start:end]

                        risks_chunk, _ = model(traces_chunk, masks_chunk)

                        g_cases, g_controls = sample_cases_controls(risks_chunk, events_chunk, durations_chunk,
                                                                    N_CONTROLS)
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
                    risks, _ = model(traces, masks)

                    g_cases, g_controls = sample_cases_controls(risks, events, durations, N_CONTROLS)

                    loss = loss_fn(g_cases, g_controls)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()

                    epoch_risks.append(risks.detach().cpu().view(-1))
                    epoch_events.append(events.detach().cpu().view(-1))
                    epoch_durations.append(durations.detach().cpu().view(-1))

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Fold {fold}, Epoch {epoch + 1}/{NUM_EPOCHS}, Training loss: {avg_train_loss:.4f}")
            writer.add_scalar("Train loss", avg_train_loss, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Learning Rate", current_lr, epoch)

            scheduler.step()

            # Training C-index
            epoch_risks = torch.cat(epoch_risks, dim=0)
            epoch_events = torch.cat(epoch_events, dim=0)
            epoch_durations = torch.cat(epoch_durations, dim=0)
            concordance_train = cindex(estimate=epoch_risks, event=epoch_events, time=epoch_durations)
            print(f"Fold {fold}, Epoch {epoch + 1}/{NUM_EPOCHS}, Training C-index: {concordance_train:.4f}")
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

                    val_risks, _ = model(val_traces, val_masks)

                    val_g_cases, val_g_controls = sample_cases_controls(val_risks, val_events, val_durations,
                                                                        N_CONTROLS)

                    val_loss += loss_fn(val_g_cases, val_g_controls).item()

                    val_epoch_risks.append(val_risks.detach().cpu().view(-1))
                    val_epoch_events.append(val_events.detach().cpu().view(-1))
                    val_epoch_durations.append(val_durations.detach().cpu().view(-1))

                avg_val_loss = val_loss / len(val_dataloader)
                print(f"Fold {fold}, Epoch {epoch + 1}/{NUM_EPOCHS}, Validation loss: {avg_val_loss:.4f}")
                writer.add_scalar("Validation loss", avg_val_loss, epoch)

            # Validation C-index
            val_epoch_risks = torch.cat(val_epoch_risks, dim=0)
            val_epoch_events = torch.cat(val_epoch_events, dim=0)
            val_epoch_durations = torch.cat(val_epoch_durations, dim=0)
            concordance_val = cindex(estimate=val_epoch_risks, event=val_epoch_events, time=val_epoch_durations)
            print(f"Fold {fold}, Epoch {epoch + 1}/{NUM_EPOCHS}, Validation C-index: {concordance_val:.4f}")
            writer.add_scalar("C-index evaluation", concordance_val, epoch)

        # Log best values (e.g., final validation cindex and training loss)
        writer.add_hparams(
            {"projection_nodes": PROJECTION_NODES,
             "attention_nodes": ATTENTION_NODES,
             "dropout": DROPOUT,
             "cox_regularization": COX_REGULARIZATION,
             "weight_decay": WEIGHT_DECAY,
             "batch_size": BATCH_SIZE,
             "learning_rate": LEARNING_RATE,
             "num_epochs": NUM_EPOCHS},
            {"hparam/concordance_val": concordance_val,
             "hparam/loss": avg_train_loss,
             "hparam/val_loss": avg_val_loss,
             "hparam/train_cindex": concordance_train}
        )

        writer.close()

        torch.save(model.state_dict(), os.path.join(SAVE_MODEL_PATH, f"{STUDY_NAME}_fold_{fold}.pth"))


if __name__ == "__main__":
    # Cross-validation
    folds = 3
    cross_val(folds=folds)