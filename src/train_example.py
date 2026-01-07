import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import os
from dotenv import load_dotenv
import sqlalchemy

# Load secrets from .env file immediately
load_dotenv()


def get_db_connection():
    """Establishes connection to PostgreSQL for annotations."""
    uri = os.getenv("POSTGRES_CONNECTION_URI")
    if not uri:
        raise ValueError("POSTGRES_CONNECTION_URI is not set in .env")
    return sqlalchemy.create_engine(uri)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup MLflow
    # Hydra changes the working dir, so we set the URI from env vars explicitly
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # Auto-tag the run with the Hydra Job ID (useful for sweeps)
    mlflow.set_experiment("My Hydra Experiment")

    with mlflow.start_run():
        # Log all Hydra parameters to MLflow
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        print(f"Training with Data Path: {os.getenv('DATASET_DIR')}")

        # Example: Fetch metadata from Postgres
        try:
            engine = get_db_connection()
            with engine.connect() as conn:
                print("Connected to PostgreSQL for metadata...")
                # result = conn.execute(sqlalchemy.text("SELECT count(*) FROM annotations"))
        except Exception as e:
            print(f"DB Connection skipped/failed: {e}")

        # --- DUMMY TRAINING LOOP ---
        # Access params via dot notation: cfg.learning_rate
        for epoch in range(cfg.epochs):
            # Simulation of training
            val_accuracy = 0.5 + (epoch * 0.01) + (cfg.learning_rate * 10)

            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        # 2. Return metric for Optuna
        # If running a sweep, Optuna needs a return value to optimize
        return val_accuracy


if __name__ == "__main__":
    main()