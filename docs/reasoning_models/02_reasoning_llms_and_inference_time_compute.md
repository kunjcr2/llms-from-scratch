# Summary: Reasoning-Based Large Language Models — Lecture 2

## Key Topics Covered

### 1. **Background on Reasoning LLMs**

- GPT-3.5 was designed to answer like humans but didn’t truly reason step-by-step.
- Since 2022, models like OpenAI’s O1 and DeepSeek R1 introduced actual reasoning abilities.
- Reasoning LLMs provide step-by-step thought processes behind answers, increasing trust and accuracy.

### 2. **Inference Time Compute Scaling**

- Humans give better answers when they spend more time thinking.
- Similarly, if LLMs “think more” (use more computational resources during inference), their accuracy improves on complex problems.
- Example: Comparing a regular LLM that answers immediately vs. one that generates step-by-step reasoning (Chain of Thought).
- Reasoning LLMs use significantly more tokens during inference to think through problems.

### 3. **Chain of Thought Prompting**

- A technique where LLMs are prompted with intermediate reasoning steps before the final answer.
- This significantly improves accuracy on arithmetic, commonsense, and symbolic reasoning tasks.
- Large models benefit most from Chain of Thought prompting, as reasoning is an emergent ability with scale.

### 4. **Zero-Shot Reasoning**

- Even without example prompts, telling the model to “think step-by-step” (zero-shot chain of thought) can trigger reasoning.
- This approach outperforms regular zero-shot prompting.

### 5. **Model Size and Reasoning**

- Larger models (e.g., PaLM with hundreds of billions of parameters) reason better than smaller models.
- Smaller models struggle to produce accurate step-by-step reasoning even with prompting.

### 6. **Hands-On Projects Preview**

- Exploring how model size affects reasoning ability on datasets like GSM-8K.
- Comparing few-shot prompting, few-shot Chain of Thought, and zero-shot Chain of Thought.
- Experimenting with models built from scratch and assessing their reasoning performance.

---

## Takeaway

- Reasoning in LLMs can be induced at inference time by using Chain of Thought prompting and allowing models to spend more compute time “thinking.”
- Reasoning is a capability that emerges strongly as model size increases.
- Techniques like zero-shot Chain of Thought provide simple but powerful ways to enhance reasoning without additional training.
