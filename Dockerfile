# Use an official PyTorch runtime with CUDA 12.1 support
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for DGL (GraphBolt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libtbb2 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Set backend to PyTorch explicitly and disable GraphBolt
ENV DGLBACKEND=pytorch
ENV DGL_GRAPH_BOLT_DISABLE=1
ENV DGL_ENABLE_GRAPHBOLT=0

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install dependencies
# Note: The base image already contains PyTorch 2.2.1 with CUDA 12.1.
# We explicitly install DGL compatible with PyTorch 2.2 and CUDA 12.1
RUN pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
# Install other dependencies, skipping torch since it's in the base image
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Set the entrypoint to run the application
ENTRYPOINT ["python", "run.py"]

# Default command can be overridden, e.g., to specify a different config
CMD ["--config", "configs/elliptic.yaml"]
