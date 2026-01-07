# Base image with CUDA support
FROM nvidia/cuda:12.4-cudnn-devel-ubuntu24.04

# Install system basics & Micromamba
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget bzip2 ca-certificates libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

RUN wget -qO- [https://micro.mamba.pm/api/micromamba/linux-64/latest](https://micro.mamba.pm/api/micromamba/linux-64/latest) | \
    tar -xvj bin/micromamba --strip-components=1 -C /usr/local

# Create environment from file (Cached layer)
COPY environment.yml /tmp/env.yml
RUN micromamba create -n base -f /tmp/env.yml && micromamba clean -a -y

# Activate environment for all future commands
SHELL ["micromamba", "run", "-n", "base", "/bin/bash", "-c"]

# Inject MLflow Config (These will be overridden by run commands, but good defaults)
ENV MLFLOW_TRACKING_INSECURE_TLS="true"

# Copy Application Code
COPY . /app
WORKDIR /app

# Default command
CMD ["python", "train.py"]