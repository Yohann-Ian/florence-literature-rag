# Dockerfile — Florence Literature RAG Agent
# Builds a CPU-only image that runs the FastAPI service.

# Start from an official slim Python base (Linux underneath).
FROM python:3.12-slim

# Avoid interactive prompts during apt installs, and unbuffer Python logs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system libraries some Python packages need (e.g. for builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than the CUDA build).
# This keeps the image from pulling gigabytes of GPU libraries it can't use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# Copy the application code into the image
COPY . .

# The container listens on port 8000
EXPOSE 8000

# Start the FastAPI service with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]