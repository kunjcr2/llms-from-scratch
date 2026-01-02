from vllm import LLM, SamplingParams
import time

# ==============================================================================
# 1. BASIC OFFLINE INFERENCE
# ==============================================================================
def run_basic_inference():
    """
    Demonstrates the standard way to load a model and generate text.
    Uses 'facebook/opt-125m' for a quick lightweight demo.
    """
    print("\n--- Starting Basic Inference ---")
    
    # 1. Define prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # 2. Define sampling parameters (temperature, top_p, etc.)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

    # 3. Load the model
    # Note: For larger models, ensure you have enough GPU VRAM.
    llm = LLM(model="facebook/opt-125m")

    # 4. Generate outputs
    outputs = llm.generate(prompts, sampling_params)

    # 5. Print output
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# ==============================================================================
# 2. SPECULATIVE DECODING
# ==============================================================================
def run_speculative_decoding():
    """
    Demonstrates Speculative Decoding implementation.
    Requires a target model and a smaller draft model.
    """
    print("\n--- Starting Speculative Decoding Demo ---")
    
    prompts = ["The future of artificial intelligence involves"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # To use speculative decoding, you specify `speculative_model`
    # Ideally: Target = Large (Llama-2-70b), Draft = Small (Llama-2-7b-Chat)
    # Here using OPT-350m (Target) and OPT-125m (Draft) as examples.
    llm = LLM(
        model="facebook/opt-350m",
        speculative_model="facebook/opt-125m",
        num_speculative_tokens=5, # Number of tokens the draft model proposes
    )

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")
    for output in outputs:
        print(f"Generated: {output.outputs[0].text}")

# ==============================================================================
# 3. AUTOMATIC PREFIX CACHING
# ==============================================================================
def run_prefix_caching_demo():
    """
    Demonstrates how to enable Automatic Prefix Caching to save memory/compute
    when prompts share a long common prefix.
    """
    print("\n--- Starting Prefix Caching Demo ---")
    
    # A long system prompt that is repeated
    system_prompt = "You are a helpful AI assistant. You answer questions concisely. " * 50
    
    prompts = [
        system_prompt + "What is the capital of France?",
        system_prompt + "What is the capital of Germany?",
        system_prompt + "What is the capital of Italy?"
    ]
    
    sampling_params = SamplingParams(temperature=0.0)

    # Enable prefix caching with `enable_prefix_caching=True`
    llm = LLM(
        model="facebook/opt-125m", 
        enable_prefix_caching=True
    )

    # First request: Prefill (caches the prefix)
    start = time.time()
    llm.generate(prompts[0], sampling_params)
    print(f"First request (Cold Cache): {time.time() - start:.4f}s")

    # Subsequent requests: Should be faster (if sequence is long enough)
    start = time.time()
    llm.generate(prompts[1:], sampling_params)
    print(f"Subsequent requests (Warm Cache): {time.time() - start:.4f}s")

# ==============================================================================
# 4. API SERVER CLIENT (EXAMPLE)
# ==============================================================================
# To use this, first start the server in your terminal:
# !vllm serve facebook/opt-125m
#
# Then run this client code:
#
# from openai import OpenAI
# 
# client = OpenAI(
#     base_url="http://localhost:8000/v1",
#     api_key="EMPTY",
# )
# 
# completion = client.chat.completions.create(
#     model="facebook/opt-125m",
#     messages=[
#         {"role": "user", "content": "Hello, explain quantum mechanics in one sentence."}
#     ]
# )
# 
# print("Server Response:", completion.choices[0].message.content)


if __name__ == "__main__":
    # Uncomment the function you want to run.
    # Note: vLLM initializes CUDA kernels globally, so it is recommended 
    # to run only one of these functions per script execution.

    run_basic_inference()
    
    # run_speculative_decoding()
    
    # run_prefix_caching_demo()
