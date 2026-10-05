FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# System setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip git curl wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set python & pip aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Python packages
RUN pip install --upgrade pip
RUN pip install "vllm[serve]" tiktoken accelerate

# Create working directory
WORKDIR /app

# Expose port
EXPOSE 8000

# Run vLLM server with GatorGPT
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
    "--model", "kunjcr2/GatorGPT", \
    "--tokenizer", "kunjcr2/GatorGPT", \
    "--download-dir", "/app/gatorgpt", \
    "--max-model-len", "2048", \
    "--dtype", "float32", \
    "--host", "0.0.0.0", \
    "--port", "8000"]
