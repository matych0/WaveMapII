import hydra
from omegaconf import DictConfig, OmegaConf
import optuna
import mlflow
import numpy as np
import os
from dotenv import load_dotenv

from .engine import run_training

load_dotenv()

def objective(trial, cfg: DictConfig):
    # 1. Suggest hyperparameters using Optuna
    # We create a local copy of the config to modify it for this trial
    trial_cfg = cfg.copy()

    # Example: Overriding specific nested config values
    trial_cfg.training.optimizer.lr = trial.suggest_categorical('lr', [1e-4, 5e-4, 1e-3])
    trial_cfg.training.optimizer.weight_decay = trial.suggest_categorical('wd', [1e-4, 1e-2])
    trial_cfg.training.hparams.batch_size = trial.suggest_categorical('bs', [8, 16, 32])
    trial_cfg.training.hparams.epochs = trial.suggest_int("epochs", 10, 15, step=1)
    
    run_name = f"trial_{trial.number}"
    
    # 2. Start the Trial Run (Parent for the folds)
    with mlflow.start_run(run_name=run_name, nested=True) as trial_run:
        # Log trial-level params
        mlflow.log_params(trial.params)
        
        # 3. Execute Cross-Validation training
        # engine.run_training returns: List[Dict{"history": ..., "final_val_cindex": ...}]
        results = run_training(trial_cfg)
        
        fold_cindexes = []

        # 4. Log each fold as a nested run under the trial
        for i, fold_data in enumerate(results):
            with mlflow.start_run(run_name=f"trial_{trial.number}_fold_{i}", nested=True):
                mlflow.set_tag("fold", i)
                
                # Log fold history (metrics over epochs)
                history = fold_data["history"]
                epochs = len(history["train_loss"])
                for step in range(epochs):
                    mlflow.log_metric("train_loss", history["train_loss"][step], step=step)
                    mlflow.log_metric("val_loss", history["val_loss"][step], step=step)
                    mlflow.log_metric("val_cindex", history["val_cindex"][step], step=step)
                
                # Log the final score of this specific fold
                mlflow.log_metric("final_fold_cindex", fold_data["final_val_cindex"])
                fold_cindexes.append(fold_data["final_val_cindex"])

        # 5. Log aggregate trial metric
        mean_cindex = np.mean(fold_cindexes)
        mlflow.log_metric("mean_val_cindex", mean_cindex)
        
        # Report back to Optuna for pruning/optimization
        return mean_cindex
    

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"{cfg.study_name}_{cfg.experiment_name}")

    # Start a parent run to group all trials
    with mlflow.start_run(run_name="bayesian_optimization") as parent_run:
        
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(multivariate=True, seed=cfg.seed),
            pruner=optuna.pruners.ThresholdPruner(lower=0.5)
        )

        # Pass the Hydra cfg to the objective function using a lambda or partial
        study.optimize(lambda trial: objective(trial, cfg), n_trials=3)

        print(f"Best value: {study.best_value} (params: {study.best_params})")
        
        # Log best params to the parent run
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mean_cindex", study.best_value)

if __name__ == "__main__":
    main()