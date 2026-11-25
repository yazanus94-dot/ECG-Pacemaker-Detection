# Modern Dockerfile for ECG Pacemaker Detection
# Uses CUDA 12.1 + Python 3.10 for broad GPU compatibility

FROM nvidia/cuda:12.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    ca-certificates \
    libpng-dev \
    libjpeg-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY README.md .
COPY LICENSE .

# Create demo data directory
RUN mkdir -p /app/demo_data

# Generate synthetic demo data at build time (optional)
# RUN python src/generate_synthetic_data.py /app/demo_data 20

# Expose port for potential API service
EXPOSE 8000

# Default command
CMD ["python", "-c", "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"]