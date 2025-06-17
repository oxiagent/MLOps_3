# Single-stage YOLOv8 SageMaker container (CPU only)
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.5.1-cpu-py311-ubuntu22.04-sagemaker

# Set environment variables to disable GPU and optimize container
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CUDA=0 \
    TORCH_CUDA_ARCH_LIST=None \
    SAGEMAKER_CONTAINER=1 \
    MONITOR_INTERVAL=30 \
    LOG_LEVEL=info \
    DISABLE_GPU=True \
    CUDA_VISIBLE_DEVICES="-1"

# Working directory for SageMaker
WORKDIR /opt/program

# Create necessary directories
RUN mkdir -p /opt/ml/input/data \
    && mkdir -p /opt/ml/output/data \
    && mkdir -p /opt/ml/model

# Update packages and install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import torch; print(f'PyTorch {torch.__version__} installed with CUDA={torch.cuda.is_available()}')" \
    && python -c "import ultralytics; print(f'Ultralytics {ultralytics.__version__} installed successfully')" \
    && rm -rf /tmp/* \
    && rm -rf /root/.cache \
    && find /opt/conda -name "*.pyc" -delete || true \
    && find /opt/conda -name "__pycache__" -delete || true

# Copy application code
COPY app.py serve monitor.py ./
RUN chmod +x serve

# Expose port 8080 for SageMaker
EXPOSE 8080 8081

# Final cleanup to minimize image size
RUN rm -rf /tmp/* /var/tmp/* /root/.cache ~/.cache \
    && apt-get update \
    && apt-get autoremove -y \
    && apt-get clean \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && find /opt -name "__pycache__" -delete || true \
    && find /opt -name "*.pyc" -delete || true \
    && du -sh / 2>/dev/null || echo "Image size check unavailable"

# Label the image
LABEL maintainer="MLOps" \
      description="YOLOv8 inference container for SageMaker (CPU only)" \
      version="1.0"

# Define entry point for SageMaker
ENTRYPOINT ["./serve"] 