# infer_mlflow.py

import os
import mlflow
import torch
import pandas as pd
import numpy as np

from omegaconf import OmegaConf
import hydra

# reuse your training utilities
from .engine import build_datasets, to_device


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def get_run_by_name(experiment_name, run_name):
    exp = mlflow.get_experiment_by_name(experiment_name)

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )

    if len(runs) == 0:
        raise ValueError(f"No run found with name {run_name}")

    return runs.iloc[0]["run_id"]


def get_child_runs(experiment_name, parent_run_id):
    exp = mlflow.get_experiment_by_name(experiment_name)

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
    )

    return runs


def load_model_from_run(run_id, cfg, device):
    # download model
    model_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
    )

    # find .pth file
    files = os.listdir(model_path)
    model_file = [f for f in files if f.endswith(".pth")][0]
    model_path = os.path.join(model_path, model_file)

    # rebuild model
    ModelClass = hydra.utils.get_class(cfg.model.model_class)
    model = ModelClass(**cfg.model.hparams).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def fix_none_strings(obj):
    if isinstance(obj, dict):
        return {k: fix_none_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_none_strings(v) for v in obj]
    elif obj == "None":
        return None
    else:
        return obj


# -------------------------------------------------------------
# Per-sample loss (IMPORTANT)
# -------------------------------------------------------------

def compute_sample_losses(task, outputs, durations, events):
    """
    Assumes your task can compute loss per sample.
    If not, we fallback to approximating by looping.
    """

    losses = []

    for i in range(len(durations)):
        loss = task.compute_loss(
            outputs[i:i+1],
            durations[i:i+1],
            events[i:i+1]
        )
        if loss is not None:
            losses.append(loss.item())
        else:
            losses.append(np.nan)

    return losses


# -------------------------------------------------------------
# Main inference
# -------------------------------------------------------------

def run_inference(experiment_name, run_name, output_csv):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- get parent run ----
    parent_run_id = get_run_by_name(experiment_name, run_name)
    print(f"Parent run_id: {parent_run_id}")

    # ---- load config ----
    config_path = mlflow.artifacts.download_artifacts(
        run_id=parent_run_id,
        artifact_path="config.json"
    )
    cfg = OmegaConf.load(config_path)

    #cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    cfg = OmegaConf.create(fix_none_strings(OmegaConf.to_container(cfg)))

    if "pos_weight" in cfg.training.loss:
        cfg.training.loss.pos_weight = None

    # ---- get child runs (folds) ----
    child_runs = get_child_runs(experiment_name, parent_run_id)

    all_results = []

    for _, row in child_runs.iterrows():
        run_id = row["run_id"]
        fold = int(row["tags.fold"])

        print(f"Processing fold {fold}, run_id={run_id}")

        # ---- load model ----
        model = load_model_from_run(run_id, cfg, device)

        # ---- dataset ----
        _, val_dataset = build_datasets(cfg, fold)

        if cfg.data.dataset.get("collate_fn"):
            collate_fn = hydra.utils.get_object(cfg.data.dataset.collate_fn)
        else:
            collate_fn = None

        DataloaderClass = hydra.utils.get_class(cfg.data.dataset.dataloader.class_name)

        val_loader = DataloaderClass(
            val_dataset,
            batch_size=cfg.training.hparams.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # ---- task ----
        TaskClass = hydra.utils.get_class(cfg.training.task.handler)
        task = TaskClass(cfg, device)

        # ---- inference ----
        model.eval()

        with torch.no_grad():
            sample_idx = 0

            for batch in val_loader:
                study_ids = batch.pop("study_id")
                center_ids = batch.pop("center_id")

                batch = to_device(batch, device)

                outputs = model(batch)
                probs = torch.sigmoid(outputs)

                events = batch["y"][:, 0].bool()
                durations = batch["y"][:, 1]

                losses = compute_sample_losses(task, outputs, durations, events)

                for i, loss in enumerate(losses):
                    all_results.append({
                        "study_id": study_ids[i],
                        "center_id": center_ids[i],
                        "fold": fold,
                        "duration": durations[i].item(),
                        "event": events[i].item(),
                        "probability": round(probs[i].item(), 4),
                        # "run_id": run_id,
                        # "sample_id": sample_idx + i,
                        "loss": round(loss, 4) # round to 4 decimals
                    })

                sample_idx += len(losses)

    # ---- save CSV ----
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)

    print(f"Saved results to {output_csv}")


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------

if __name__ == "__main__":

    experiment_name = "WMP2_sanity_check_binary_classification"
    run_name = "egm_resnet_attention_binary"
    output_csv = "inference_results/egm_val_losses.csv"

    mlflow.set_tracking_uri("file:./mlruns")

    run_inference(
        experiment_name=experiment_name,
        run_name=run_name,
        output_csv=output_csv
    )