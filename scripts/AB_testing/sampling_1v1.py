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
from src.dataset.collate import collate_padding_merged, collate_validation
from src.dataset.dataset import HDFDataset, ValidationDataset
from src.transforms.transforms import (BaseTransform, RandomAmplifier,
                                       RandomGaussian, RandomPolarity,
                                       RandomShift, RandomTemporalScale,
                                       TanhNormalize)

#Set seed
SEED = 3052001

ANNOTATION_DIR = ""
DATA_DIR = ""

SAVE_MODEL_PATH = ""
STUDY_NAME = "sampling_1v1"
TB_LOG_DIR = ""

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
CHUNK_SIZE = 4
FOLDS = 3

# Optimized hyperparameters
DROPOUT = 0.5
COX_REGULARIZATION = 0.01
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 16
NUM_EPOCHS = 264
N_CONTROLS = 4


def get_predictions(loader, model):
    all_risks = []
    all_durations = []
    all_events = []

    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for batch in loader:
            durations, events, traces, traces_masks = batch

            durations = torch.tensor(durations, dtype=torch.float32)
            events = torch.tensor(events, dtype=torch.bool)

            durations = durations.to(device)
            events = events.to(device)
            traces = traces.to(device)
            traces_masks = traces_masks.to(device)

            risks, _ = model(traces, traces_masks)

            all_risks.append(risks.view(-1))        # Instead of .squeeze()
            all_durations.append(durations.view(-1))
            all_events.append(events.view(-1))

    risks_tensor = torch.cat(all_risks).cpu()
    durations_tensor = torch.cat(all_durations).cpu()
    events_tensor = torch.cat(all_events).cpu()

    return risks_tensor, durations_tensor, events_tensor


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
        train_dataset = HDFDataset(
            annotations_file=ANNOTATION_DIR,
            data_dir=DATA_DIR,
            startswith="LA",
            train=True,
            transform=train_transform,
            readjustonce=True,
            segment_ms=SEGMENT_MS,
            filter_utilized=FILTER_UTILIZED,
            cross_val_fold=fold,
            random_seed=SEED,
        )
        
        train_cindex_dataset = ValidationDataset(
            annotations_file=ANNOTATION_DIR,
            data_dir=DATA_DIR,
            startswith="LA",
            eval_data=False,
            transform=train_transform,
            readjustonce=True,
            segment_ms=SEGMENT_MS,
            filter_utilized=FILTER_UTILIZED,
            cross_val_fold=fold,
        )

        val_cindex_dataset = ValidationDataset(
            annotations_file=ANNOTATION_DIR,
            data_dir=DATA_DIR,
            startswith="LA",
            eval_data=True,
            transform=val_transform,
            readjustonce=True,
            segment_ms=SEGMENT_MS,
            filter_utilized=FILTER_UTILIZED,
            cross_val_fold=fold
        )

        # Create DataLoader
        generator = torch.Generator()
        generator.manual_seed(SEED)
        
        val_batch_size = BATCH_SIZE if BATCH_SIZE <= CHUNK_SIZE else CHUNK_SIZE
        
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator, drop_last=True, collate_fn=collate_padding_merged)
        
        train_cindex_dataloader = DataLoader(train_cindex_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_validation)
        val_cindex_dataloader = DataLoader(val_cindex_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=collate_validation)

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
            for traces, masks in train_dataloader:
                traces = traces.to(device)
                masks = masks.to(device)
                
                # Gradient accumulation
                if BATCH_SIZE > CHUNK_SIZE:

                    accumulation_steps = BATCH_SIZE // CHUNK_SIZE
                    optimizer.zero_grad()
                    batch_loss = 0.0 
                    
                    case_traces, control_traces = torch.chunk(traces, chunks=2, dim=0)
                    case_masks, control_masks = torch.chunk(masks, chunks=2, dim=0)
                    
                    case_traces_chunks = torch.chunk(case_traces, chunks=accumulation_steps, dim=0)
                    control_traces_chunks = torch.chunk(control_traces, chunks=accumulation_steps, dim=0)
                    case_masks_chunks = torch.chunk(case_masks, chunks=accumulation_steps, dim=0)
                    contrrol_masks_chunks = torch.chunk(control_masks, chunks=accumulation_steps, dim=0)

                    for i in range(accumulation_steps):
                        
                        case_traces_chunk = case_traces_chunks[i]
                        control_traces_chunk = control_traces_chunks[i]
                        case_masks_chunk = case_masks_chunks[i]
                        contrrol_masks_chunk = contrrol_masks_chunks[i]
                        
                        traces_chunk = torch.cat([case_traces_chunk, control_traces_chunk], dim=0)
                        masks_chunk = torch.cat([case_masks_chunk, contrrol_masks_chunk], dim=0)

                        risks_chunk, _ = model(traces_chunk, masks_chunk)
                        
                        g_case, g_control = torch.chunk(risks_chunk, chunks=2, dim=0)

                        loss = loss_fn(g_case, g_control)

                        # Normalize loss to maintain consistent scale
                        loss = loss / accumulation_steps
                        loss.backward()

                        batch_loss += loss.item()

                        # Store for logging
                        epoch_risks.append(risks_chunk.detach().cpu().view(-1))

                    optimizer.step()
                    total_train_loss += batch_loss

                else:
                    
                    risks, _ = model(traces, masks)

                    g_cases, g_controls = torch.chunk(risks, chunks=2, dim=0)

                    loss = loss_fn(g_cases, g_controls)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()

                    epoch_risks.append(risks.detach().cpu().view(-1))

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Fold {fold}, Epoch {epoch+1}/{NUM_EPOCHS}, Training loss: {avg_train_loss:.4f}")
            writer.add_scalar("Train loss", avg_train_loss, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Learning Rate", current_lr, epoch)

            scheduler.step()

            # Training C-index
            train_preds , train_durations, train_events = get_predictions(train_cindex_dataloader, model)
            concordance_train = cindex(estimate=train_preds.view(-1), event=train_events.view(-1), time=train_durations.view(-1))
            print(f"Fold {fold}, Epoch {epoch+1}/{NUM_EPOCHS}, Training C-index: {concordance_train:.4f}")
            writer.add_scalar("C-index training", concordance_train, epoch)
            
            # Validation C-index
            val_preds , val_durations, val_events = get_predictions(val_cindex_dataloader, model)
            concordance_val = cindex(estimate=val_preds.view(-1), event=val_events.view(-1), time=val_durations.view(-1))
            print(f"Fold {fold}, Epoch {epoch+1}/{NUM_EPOCHS}, Validation C-index: {concordance_val:.4f}")
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
            "hparam/train_cindex": concordance_train}
        )

        writer.close()

        torch.save(model.state_dict(), os.path.join(SAVE_MODEL_PATH, f"{STUDY_NAME}_fold_{fold}.pth"))


if __name__ == "__main__":
    # Cross-validation
    folds = 3
    cross_val(folds=folds)