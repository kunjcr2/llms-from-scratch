# **Loading pretrained weights**

- You can download the weights from `https://www.kaggle.com/datasets/xhlulu/openai-gpt2-weights?select=124M`

Whenever we downlaod those 7 files from lets say, kaggle. We have a function `load_gpt2_params_from_tf_ckpt()` in the script `gpt_download3.py`, which returns a disctionary of parameters. And it has 5 keys:

1. wte
   - Token Embeddings
2. wpe
   - Positional Embeddings
3. Blocks
   - Attention blocks with Q, K, V
   - Feed forward neural network weights
   - Output projection which gives logits
   - Layer normalization scale and shift for both of the kinds
     > Similar for all transformer blocks
4. Final normalization scale
5. Final normalization shift

---

There is basically a class and weights to be downloaded from internet or from kaggle which, by using some sort of processing, and then we can use those.

---

Go to `https://youtu.be/yXrGeDNuymY?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu`.

- This is a bit complex, you can go the video and get the idea of what are we doing. This is much complex than what I expected it to be. I am so sorry for being lazy !

---

# **Fine tune**

- Adapting a pretrained model to a specific task by training the model on additional data is what we call as fintuning.

Two types of the finetuning:

1.  Instruction finetuning, here we give the LLM an instruction and which can be then applied to the main body to perform tasks like current GPT does.
2.  Classification finetuning, and here we basically just restict the model to answer in few classifications. Like email spam or not spam. AND, i dont care about you. Coming back to instructions.

There are 2 ways or methods for finetuning. Lora and QLore. Read about them from `https://www.redhat.com/en/topics/ai/lora-vs-qlora#:~:text=QLoRA%20and%20LoRA%20are%20both%20fine-tuning%20techniques%20that,trains%20the%20parameters%20necessary%20to%20learn%20new%20information.` Dont go much in this. For now, just remember that these are just better ways of finetuning the models.

---

---

---

**I am skipping a big chunkn of lectures as they are classification fine tuning. I have seen those videos but not coded out. Here are the links to it:**

Starts from here, till lecture 37.
`https://www.youtube.com/watch?v=yZpy_hsC1bE&list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu&index=33&pp=iAQB0gcJCY0JAYcqIYzv`

The only thing in the video is classification LLMs have a small change and that is, we had to move from logits being of the size 50257, we just needed it to be size of 2 (or size of outcomes in classification.)

---

---

---

Now from the next one, we actually get the stuff. We call it **Instruction finetuning**.
