# build_and_run.ps1
param (
    [string]$ConfigName = "elliptic.yaml"
)

# --- Build the Docker Image ---
Write-Host "Building Docker image 'fraud-detection-gnn'..."
docker build -t fraud-detection-gnn .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed. Exiting."
    exit 1
}

# --- Run the Docker Container ---
Write-Host "Running container with config: $ConfigName"
# The command inside the container will be `python run.py --config configs/elliptic.yaml`
docker run --rm fraud-detection-gnn --config "configs/$ConfigName"
