# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use a DGL/PyTorch combination known to be stable
RUN pip install --no-cache-dir torch==2.2.1
RUN pip install --no-cache-dir dgl==2.1.0 -f https://data.dgl.ai/wheels/repo.html
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Set the entrypoint to run the application
ENTRYPOINT ["python", "run.py"]

# Default command can be overridden, e.g., to specify a different config
CMD ["--config", "configs/elliptic.yaml"]
