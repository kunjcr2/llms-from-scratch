FROM vllm/vllm-openai:latest

COPY model/ /model/

EXPOSE 8000

CMD ["vllm", "serve", "/model", "--host", "0.0.0.0", "--port", "8000", "--trust-remote-code", "--enable-lora", "--lora-modules", "dpo=/model/dpo_adapters", "--max-lora-rank", "64", "--max-model-len", "4096", "--gpu-memory-utilization", "0.90"]
