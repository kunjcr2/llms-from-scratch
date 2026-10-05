# GatorGPT

GatorGPT is a lightweight transformer-based language model trained on the TinyStories dataset. It incorporates several modern architectural choices for efficient training and inference.

## Model Architecture

- **Base Architecture**: Transformer decoder-only
- **Model Size**: 384-dimensional embeddings (d_model)
- **Attention**: Grouped Query Attention (GQA) with 8 heads and 2 KV groups
- **Positional Encoding**: Rotary Positional Encoding (RoPE)
- **Feed Forward**: SwiGLU activation with ~2.5x expansion ratio
- **Normalization**: RMSNorm
- **Vocabulary**: p50k_base (50,257 tokens)
- **Context Length**: 1024 tokens
- **Parameters**: 10 transformer blocks

## Training Metrics

- **Training Data**: 214,198,685 tokens from TinyStories dataset
- **Validation Data**: 11,310,150 tokens
- **Training Batch Size**: 16
- **Sequence Length**: 512
- **Learning Rate**: 3e-4
- **Optimizer**: AdamW with weight decay 0.01

## Inference

The model is deployed using vLLM for efficient inference with an OpenAI-compatible API interface.

### Using Docker

```bash
# Pull and run the Docker container
docker run -d --gpus all -p 8000:8000 kunjcr2/gatorgpt
```

### API Usage

Once the server is running, you can use it like any OpenAI-compatible API:

```python
import requests
import json

def generate_text(prompt, max_tokens=50):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    response = requests.post(
        "http://localhost:8000/v1/completions",
        headers=headers,
        data=json.dumps(data)
    )

    return response.json()["choices"][0]["text"]

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)
```

### Server Configuration

The model server uses the following configuration:

- Maximum sequence length: 2048 tokens
- Data type: float32
- Host: 0.0.0.0 (accessible from any network interface)
- Port: 8000

## Model Links

🤗 HuggingFace: https://huggingface.co/kunjcr2/GatorGPT

🐋 Docker Hub: https://hub.docker.com/repository/docker/kunjcr2/gatorgpt/general

## Features

- **Fast Inference**: Optimized with torch.compile and flash attention
- **Memory Efficient**: Uses Grouped Query Attention to reduce memory usage
- **BF16 Support**: Native bfloat16 support for A100 GPUs
- **Efficient Training**: Parallel data processing and optimized dataloaders
