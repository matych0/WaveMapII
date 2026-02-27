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
    
    #tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    #mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment(f"{cfg.study_name}_{cfg.experiment_name}")
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name=f"{cfg.run_name}") as parent_run:
        print(f"Run ID: {parent_run.info.run_id}")

        mlflow.log_params({
            "learning_rate": cfg.training.optimizer.lr,
            "weight_decay": cfg.training.optimizer.weight_decay,
            "batch_size": cfg.training.hparams.batch_size,
            "epochs": cfg.training.hparams.epochs,
        })

        mlflow.set_tag("model_type", cfg.model.name) # Add a tag for easy filtering

        mlflow.log_dict(cfg, "config.json")

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

                history = fold["history"]

                for j in range(cfg.training.hparams.epochs):
                    for metric_name, values in history.items():
                        if j < len(values) and values[j] is not None:
                            mlflow.log_metric(metric_name, float(values[j]), step=j)

        # log average
        val_metric_name = None
        for k in results[0]["history"].keys():
            if k.startswith("val_") and k != "val_loss":
                val_metric_name = k
                break

        if val_metric_name is not None:
            final_vals = []
            for f in results:
                vals = f["history"][val_metric_name]
                final_vals.append(vals[-1])
            mlflow.log_metric(f"{val_metric_name}_mean", sum(final_vals) / len(final_vals))

        return results


if __name__ == "__main__":
    main()
