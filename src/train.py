# src/train.py
from dotenv import load_dotenv
import os
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch


from .engine import run_training
from .utils.seed import set_seed

load_dotenv()

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    #set_seed(cfg.seed)
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(f"{cfg.study_name}_{cfg.experiment_name}")
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name=f"{cfg.model.name}") as parent_run:
        print(f"Run ID: {parent_run.info.run_id}")

        mlflow.log_params({
            "learning_rate": cfg.training.optimizer.lr,
            "weight_decay": cfg.training.optimizer.weight_decay,
            "batch_size": cfg.training.hparams.batch_size,
            "epochs": cfg.training.hparams.epochs,
        })

        mlflow.set_tag("model_type", cfg.model.name) # Add a tag for easy filtering

        results = run_training(cfg)

        # log fold results
        for i, fold in enumerate(results):
            with mlflow.start_run(run_name=f"Fold_{i+1}", nested=True) as child_run:
                print(f"Run ID: {child_run.info.run_id}")
                mlflow.set_tag("fold", i + 1)

                """ mlflow.pytorch.log_model(
                    pytorch_model=fold["model"],
                    artifact_path="model",
                    registered_model_name=f"{cfg.model.name}"  # optional but recommended
                ) """

                for j in range(cfg.training.hparams.epochs):
                    mlflow.log_metric(f"train_loss", fold["history"]["train_loss"][j], step=j)
                    mlflow.log_metric(f"val_loss", fold["history"]["val_loss"][j], step=j)
                    mlflow.log_metric(f"train_cindex", fold["history"]["train_cindex"][j], step=j)
                    mlflow.log_metric(f"val_cindex", fold["history"]["val_cindex"][j], step=j)

        # log average
        avg_cidx = sum(f["final_val_cindex"] for f in results) / len(results)
        mlflow.log_metric("cindex_mean", avg_cidx)

        return results


if __name__ == "__main__":
    main()
