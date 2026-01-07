# Reads .env for local use
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Names
PROJECT_NAME = my-project
DOCKER_TAG = $(PROJECT_NAME):latest

.PHONY: help setup update train clean docker-build docker-run

help:
	@echo "🛠️  $(PROJECT_NAME) Makefile"
	@echo "---------------------------"
	@echo "make setup        - Create local Mamba environment"
	@echo "make update       - Update local Mamba environment from environment.yml"
	@echo "make train        - Run local training (Hydra)"
	@echo "make sweep        - Run Optuna hyperparameter sweep"
	@echo "make clean        - Remove pycache, hydra outputs, and local logs"
	@echo "make docker-build - Build the Docker image"
	@echo "make docker-run   - Run the container on Server (mounts ZFS)"

# --- Local Development ---
setup:
	mamba env create -f environment.yml

update:
	mamba env update -f environment.yml --prune

train:
	python src/train.py

sweep:
	python src/train.py --multirun hydra/sweeper=basic_sweep

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf outputs/ multirun/ mlruns/ .pytest_cache/

# --- Server / Docker ---
docker-build:
	docker build -t $(DOCKER_TAG) .

# Tento příkaz simuluje serverové prostředí:
# 1. Připojí GPU
# 2. Sdílí síť (pro MLflow)
# 3. Připojí ZFS dataset (změňte cestu podle potřeby)
docker-run:
	docker run --rm -it \
		--gpus all \
		--network="host" \
		--ipc=host \
		--env-file .env \
		-v /srv/datasets/my_project:/app/data/full \
		-e DATASET_DIR=/app/data/full \
		$(DOCKER_TAG)