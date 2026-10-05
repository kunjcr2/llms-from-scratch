# Use CUDA base image (for L4/A100 GPUs)
FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Use bash instead of sh
SHELL ["/bin/bash", "-c"]

# Set noninteractive frontend and timezone (US)
ENV DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles

# System setup
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python
RUN ln -sf $(which python3) /usr/bin/python
RUN /usr/bin/python --version

# Upgrade pip and install vLLM
RUN pip install --upgrade pip setuptools wheel

# Avoid the broken tokenizers sdist; install a prebuilt wheel
RUN pip install --no-cache-dir --only-binary=:all: "tokenizers==0.20.1"

# Now install vLLM normally
RUN pip install --no-cache-dir vllm

# Expose port 8000
EXPOSE 8000

# Run vLLM OpenAI-compatible API server
CMD vllm serve kunjcr2/llama3-3b-lora-openhermes \
    --dtype float16 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 8192
