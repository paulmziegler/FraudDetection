# Docker Build and Run Script for Fraud Detection

Write-Host "--- Starting Docker Build and Run Process ---" -ForegroundColor Cyan

# 1. Build the Docker Image
Write-Host "`n[1/2] Building Docker image..." -ForegroundColor Yellow
docker-compose build

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError: Docker build failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

# 2. Run the Docker Container
# Note: This will use the GPU configuration defined in docker-compose.yml
Write-Host "`n[2/2] Running Docker container..." -ForegroundColor Yellow
Write-Host "Training process will begin inside the container.`n" -ForegroundColor Gray
docker-compose up

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nError: Docker run failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n--- Docker Process Completed Successfully ---" -ForegroundColor Green
