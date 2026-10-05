import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, LoraConfig
import gradio as gr

# Load base model
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Load LoRA adapter (from local folder containing safetensor + json)
adapter_path = "/content/" 
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Gradio interface
def generate(prompt): 
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.Interface(fn=generate, inputs="text", outputs="text", title="FLAN-T5 StackOverflow Assistant").launch()