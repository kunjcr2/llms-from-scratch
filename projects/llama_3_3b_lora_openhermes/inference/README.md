# Llama-3.2-3B Fine-tuned on OpenHermes

🚀 **Instruction-tuned Llama-3.2-3B** using LoRA adapters on the [OpenHermes dataset](https://huggingface.co/datasets/teknium/OpenHermes). Deployed with **vLLM** for optimized inference.

## 📌 Model Details

- **Base model**: Meta’s Llama-3.2-3B (non-instruct)
- **Fine-tuning method**: LoRA (24.3M trainable params ≈ 0.75%)
- **Frameworks**: Hugging Face Transformers, PEFT, Accelerate
- **Inference**: vLLM runtime for high-performance serving
- **Training setup**: bf16 precision, gradient checkpointing, trained on A100 GPU

## 📊 Results

- Training loss: **1.27 → 0.20**
- Training steps: ~2000 (~4.5 hours on A100)
- Updated parameters: **0.75% of total (24.3M / 3.24B)**

## 🛠 Deployment

This model is packaged into a reproducible Docker image with **vLLM** for high-performance inference.

- **Docker Hub**: [`kunjcr2/llama-3.2-3b-openhermes`](https://hub.docker.com/r/kunjcr2/llama-3.2-3b-vllm)
- **Run locally:**

```bash
docker pull kunjcr2/llama-3.2-3b-openhermes
docker run -p 8000:8000 kunjcr2/llama-3.2-3b-openhermes
```

- Exposes OpenAI-compatible API endpoints via vLLM:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kunjcr2/llama-3.2-3b-openhermes",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

🌍 Impact

This project shows how to transform a raw base LLM into an instruction-following assistant, fine-tune efficiently with LoRA, and deploy as a scalable Dockerized service using vLLM.
