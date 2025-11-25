import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.dataset.dataset import HDFDataset, ValidationDataset
from src.dataset.collate import collate_padding, collate_validation
from model.cox_mil_resnet import CoxAttentionResnet
from losses.loss import CoxLoss
from src.transforms.transforms import (BaseTransform, RandomAmplifier, RandomGaussian,
                                       RandomTemporalScale, RandomShift, TanhNormalize)
import datetime
import numpy as np
from torchsurv.metrics.cindex import ConcordanceIndex
from statistics import mean


#Set seed
SEED = 3052001

ANNOTATION_DIR = "/media/guest/DataStorage/WaveMap/HDF5/annotations_train.csv"
DATA_DIR = "/media/guest/DataStorage/WaveMap/HDF5"

FOLD = 3

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


def get_predictions(loader, model):
    all_risks = []
    all_durations = []
    all_events = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            durations, events, traces, traces_masks = batch

            durations = torch.tensor(durations, dtype=torch.float32)
            events = torch.tensor(events, dtype=torch.bool)

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


def cross_val():
    """Cross-validation function to evaluate the model."""

    # Hyperparameters to optimize
    projection_nodes = 128
    attention_nodes = 64
    dropout = 0.5
    cox_regularization = 0.01
    learning_rate = 0.001
    weight_decay = 0.001
    batch_size =  24
    num_epochs =  413


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

    for i in range(2):

        print(f"Fold {FOLD}, Run {i}")

        set_seed(SEED)

        # Transformations
        temporal_scale = RandomTemporalScale(probability=0.2, limit=0.2, shuffle=True, random_seed=SEED)
        amplifier = RandomAmplifier(probability=0.2, limit=0.2, shuffle=True, random_seed=SEED)
        noise = RandomGaussian(probability=0.2, low_limit=10, high_limit=30, shuffle=True, random_seed=SEED)
        shift = RandomShift(probability=0.5, shift_range=0.3, shuffle=True, random_seed=SEED)
        tanh_normalize = TanhNormalize(factor=5)
        shuffle = BaseTransform(shuffle=True, random_seed=SEED)

        train_transform = transforms.Compose([
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
            train=True,
            transform=train_transform,
            startswith="LA",
            readjustonce=True,
            segment_ms=SEGMENT_MS,
            filter_utilized=FILTER_UTILIZED,
            oversampling_factor=OVERSAMPLING_FACTOR,
            cross_val_fold=FOLD,
        )

        val_dataset = ValidationDataset(
            annotations_file=ANNOTATION_DIR,
            data_dir=DATA_DIR,
            eval_data=True,
            transform=val_transform,
            startswith="LA",
            readjustonce=True,
            segment_ms=SEGMENT_MS,
            filter_utilized=FILTER_UTILIZED,
            cross_val_fold=FOLD,
        )

        """ train_cindex_dataset = ValidationDataset(
            annotations_file=ANNOTATION_DIR,
            data_dir=DATA_DIR,
            eval_data=False,
            transform=val_transform,
            startswith="LA",
            readjustonce=True,
            segment_ms=SEGMENT_MS,
            filter_utilized=FILTER_UTILIZED,
            cross_val_fold=FOLD,
        )

        val_loss_dataset = HDFDataset(
            annotations_file=ANNOTATION_DIR,
            data_dir=DATA_DIR,
            train=False,
            transform=train_transform,
            startswith="LA",
            readjustonce=True,
            segment_ms=SEGMENT_MS,
            filter_utilized=FILTER_UTILIZED,
            oversampling_factor=OVERSAMPLING_FACTOR,
            cross_val_fold=FOLD,
        ) """


        # Create DataLoader
        generator = torch.Generator()
        generator.manual_seed(SEED)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=collate_padding)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_validation)
        """ val_loss_dataloader = DataLoader(val_loss_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_padding)
        train_cindex_dataloader = DataLoader(train_cindex_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_validation) """

        # Create a directory for TensorBoard logs
        log_dir = f"runs/cross_val_reproducibility_fixed/fold_{FOLD}/run_{i}/"
        writer = SummaryWriter(log_dir)

        # Model, Loss, Optimizer
        model = CoxAttentionResnet(resnet_params, amil_params)
        loss_fn = CoxLoss()
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
            model.train()
            for case, control, case_mask, contrl_mask in train_dataloader:
                g_case, a_case = model(case, case_mask)
                g_control, a_control = model(control, contrl_mask)
                loss = loss_fn(g_case, g_control, shrink=cox_regularization)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Fold {FOLD}, run {i}, Epoch {epoch+1}/{num_epochs}, Training loss: {avg_train_loss:.4f}")
            writer.add_scalar("Train loss", avg_train_loss, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Learning Rate", current_lr, epoch)

            scheduler.step()

            """ # Validation loss
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for val_case, val_control, val_case_mask, val_contrl_mask in val_loss_dataloader:
                    val_g_case, val_a_case = model(val_case, val_case_mask)
                    val_g_control, val_a_control = model(val_control, val_contrl_mask)
                    val_loss += loss_fn(val_g_case, val_g_control, shrink=cox_regularization).item()
                avg_val_loss = val_loss / len(val_loss_dataloader)
                print(f"Fold {FOLD}, run {i}, Epoch {epoch+1}/{num_epochs}, Validation loss: {avg_val_loss:.4f}")
                writer.add_scalar("Validation loss", avg_val_loss, epoch) """

            """ # Training C-index
            train_preds , train_durations, train_events = get_predictions(train_cindex_dataloader, model)
            concordance_train = cindex(estimate=train_preds.view(-1), event=train_events.view(-1), time=train_durations.view(-1))
            print(f"Fold {FOLD}, run {i}, Epoch {epoch+1}/{num_epochs}, Training C-index: {concordance_train:.4f}")
            writer.add_scalar("C-index training", concordance_train, epoch) """

            # Validation C-index
            val_preds , val_durations, val_events = get_predictions(val_dataloader, model)
            concordance_val = cindex(estimate=val_preds.view(-1), event=val_events.view(-1), time=val_durations.view(-1))
            print(f"Fold {FOLD}, run {i}, Epoch {epoch+1}/{num_epochs}, Validation C-index: {concordance_val:.4f}")
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
            #"hparam/val_loss": avg_val_loss,
            #"hparam/train_cindex": concordance_train
            }
        )

        writer.close()

        #torch.save(model.state_dict(), f"/home/guest/lib/data/saved_models/cross_val_trial_27_fold{fold}.pth")


if __name__ == "__main__":
    # Cross-validation
    cross_val()
