# Proper SFT Pipeline for Base Model

## Data Preprocessing Functions

### 1. QA Pairs Preprocessing Function

```python
def preprocess_qa_pairs(dataset, question_field="question", answer_field="answer", prompt_template=None):
    """
    Preprocess QA pairs for base model SFT training.

    Args:
        dataset: Dataset containing QA pairs
        question_field: Field name for questions
        answer_field: Field name for answers
        prompt_template: Optional custom prompt template

    Returns:
        List of dictionaries with 'prompt' and 'completion'
    """
    processed_data = []

    for item in dataset:
        question = item[question_field]
        answer = item[answer_field]

        # Create prompt (input) and completion (target)
        if prompt_template:
            prompt = prompt_template.format(question=question)
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        processed_data.append({
            "prompt": prompt,
            "completion": answer
        })

    return processed_data
```

### 2. Chat Format Preprocessing Function

```python
def preprocess_chat_data(dataset, messages_field="messages", prompt_template=None):
    """
    Preprocess chat data for base model SFT training.

    Args:
        dataset: Dataset containing chat conversations
        messages_field: Field name containing message list
        prompt_template: Optional custom prompt template

    Returns:
        List of dictionaries with 'prompt' and 'completion'
    """
    processed_data = []

    for item in dataset:
        messages = item[messages_field]

        # Separate conversation history (prompt) from assistant response (completion)
        conversation_history = []
        assistant_response = ""

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "assistant":
                assistant_response = content
                break  # Stop at first assistant response for simplicity
            else:
                if role == "system":
                    conversation_history.append(f"System: {content}")
                else:  # user
                    conversation_history.append(f"User: {content}")

        # Create prompt from conversation history
        prompt = "\n\n".join(conversation_history) + "\n\nAssistant:"

        processed_data.append({
            "prompt": prompt,
            "completion": assistant_response
        })

    return processed_data
```

## Proper SFT Training Pipeline

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ProperSFTPipeline:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_data(self, processed_data, max_length=512):
        """Tokenize prompts and completions separately"""
        tokenized_data = []

        for item in processed_data:
            # Tokenize prompt
            prompt_tokens = self.tokenizer(
                item["prompt"],
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # Tokenize completion (target)
            completion_tokens = self.tokenizer(
                item["completion"],
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # Combine prompt and completion for full sequence
            full_text = item["prompt"] + "\n" + item["completion"]
            full_tokens = self.tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )

            # Create labels: -100 for prompt tokens, actual tokens for completion
            prompt_len = prompt_tokens["input_ids"].shape[1]
            labels = full_tokens["input_ids"].clone()
            labels[:, :prompt_len] = -100  # Ignore loss for prompt tokens

            tokenized_data.append({
                "input_ids": full_tokens["input_ids"],
                "attention_mask": full_tokens["attention_mask"],
                "labels": labels
            })

        return tokenized_data

    def calculate_loss(self, batch):
        """Calculate loss for a batch of data"""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        return outputs.loss

    def train(self, training_data, epochs=3, learning_rate=5e-5, batch_size=4):
        """Proper SFT training loop with loss calculation"""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Create data loader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for batch in dataloader:
                # Move batch to model device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                # Forward pass - model generates logits
                loss = self.calculate_loss(batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if num_batches % 10 == 0:
                    print(f"Batch {num_batches}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def generate_response(self, prompt, max_length=100):
        """Generate response for a given prompt"""
        self.model.eval()

        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=inputs["input_ids"].shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part (after the prompt)
            generated_text = response[len(prompt):].strip()

            return generated_text
```

## Complete Usage Example

```python
# Example dataset
qa_dataset = [
    {"question": "What is AI?", "answer": "Artificial Intelligence is the simulation of human intelligence processes by machines."},
    {"question": "How does ML work?", "answer": "Machine learning uses algorithms to parse data, learn from it, and make predictions."}
]

chat_dataset = [
    {
        "messages": [
            {"role": "user", "content": "What's the weather like today?"},
            {"role": "assistant", "content": "I don't have access to real-time weather data, but you can check your local weather app or website."}
        ]
    }
]

# Initialize pipeline
pipeline = ProperSFTPipeline("gpt2")  # Replace with your base model

# Preprocess data
qa_processed = preprocess_qa_pairs(qa_dataset)
chat_processed = preprocess_chat_data(chat_dataset)

# Tokenize data
qa_tokenized = pipeline.tokenize_data(qa_processed)
chat_tokenized = pipeline.tokenize_data(chat_processed)

# Combine datasets
all_training_data = qa_tokenized + chat_tokenized

# Train the model
pipeline.train(all_training_data, epochs=3, learning_rate=5e-5)

# Test generation
test_prompt = "Question: What is machine learning?\n\nAnswer:"
response = pipeline.generate_response(test_prompt)
print(f"Prompt: {test_prompt}")
print(f"Generated: {response}")
```

## Key Differences from Previous Version

1. **Proper Prompt-Completion Separation**: We clearly separate input (prompt) from target (completion)
2. **Correct Loss Masking**: Use `labels` with `-100` for prompt tokens to ignore loss on input
3. **Actual Generation**: The model generates responses during inference
4. **Loss Calculation**: Proper comparison between generated tokens and target tokens

This is the standard SFT approach where the model learns to continue the prompt with the desired completion, and we only calculate loss on the completion part.
