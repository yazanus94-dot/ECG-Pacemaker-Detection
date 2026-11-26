# CPU-only image for the ECG pacemaker-detection demo.
#
# The previous CUDA/devel base image pulled a multi-gigabyte toolchain and made
# CI builds slow and flaky. Training and inference in this repository are
# CPU-feasible, and the CPU PyTorch wheels keep the image small and the build
# dependable. Use a GPU-enabled environment directly if you need CUDA.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# build-essential: source builds for scipy-adjacent wheels; libgomp1: OpenMP
# runtime required by CPU PyTorch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU wheels first so the later requirements install does not pull the CUDA
# variant of torch.
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY README.md .
COPY LICENSE .

RUN mkdir -p /app/demo_data

# Fail the build early if the model package cannot be imported.
RUN python -c "from src.model import CTN; print('CTN import OK')"

EXPOSE 8000

CMD ["python", "-c", "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"]
