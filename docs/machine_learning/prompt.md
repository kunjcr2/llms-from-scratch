# 🧠 Ultimate Guide to Prompt Engineering (Inspired by Google’s Course)

## 🔥 What Is Prompt Engineering?

**Prompt Engineering** is the art of _asking an AI the right way_ to get what you want.

Good prompts = Clear + Structured + Contextual + Goal-oriented.  
Bad prompts = Vague + Open-ended + Unclear intent.

---

## ✋ 1. Set the **INTENT** First

> 🧾 **Before writing the prompt**, ask:
>
> - What do I want? (Answer, generation, transformation?)
> - In what format? (List, table, paragraph, code?)
> - Who is the audience?

**Examples**:

- ✘ Bad: `Tell me about dogs.`
- ✔ Good: `Write a 3-paragraph essay for a 5th grader explaining what makes dogs good pets.`

**🧠 Hack**: _Always imagine you’re talking to a brilliant but *confused intern.*_

---

## 📦 2. Use the **TCR Method**: Task → Context → Rules

### ✅ T – Task

> What exactly should the AI do?

**Example**: “Summarize this article in 3 key bullet points.”

### ✅ C – Context

> Give it background, references, or data to work with.

**Example**:  
“Based on the following text:  
`The Earth is warming due to increased CO2...`”

### ✅ R – Rules

> Add constraints or style guides.

**Example**:  
“Use a formal tone. Max 100 words. Use simple language.”

---

## 🧪 3. Use **Few-Shot Prompting** (Give Examples)

> Show the AI _how_ to respond.

**Prompt**:

```
Q: What is the capital of France?
A: Paris
Q: What is the capital of Germany?
A: Berlin
Q: What is the capital of Italy?
A:
```

→ AI learns the pattern.

Use this to train it on _custom formats_ (like how to answer tech support tickets, emails, or markdown).

---

## 🧵 4. Use **Chain-of-Thought (CoT) Prompting**

> Ask the AI to _think step-by-step_.

**Prompt**:

```
Solve this: A train leaves the station at 2 PM traveling at 60 mph...
Think step-by-step.
```

→ Helps AI solve math, logic, and reasoning tasks more accurately.

---

## 🌳 5. Try **Tree-of-Thought (ToT)**

> Instead of one answer, ask AI to explore multiple options.

**Prompt**:

```
Generate 3 different ways to solve this problem. For each, list pros and cons. Then choose the best one.
```

→ Powerful for **creative thinking**, decision-making, and product ideation.

---

## 👑 6. Role Play: Use **Persona-Based Prompts**

> Make the AI “act as” a specific expert.

**Prompt**:

```
Act as a senior UX designer. Review this UI and give me critical feedback.
```

Or…

```
You are a helpful Linux command line assistant. I will give you vague descriptions and you’ll translate them into commands.
```

→ This gets **way more accurate** responses in technical or expert tasks.

---

## 🔁 7. Iterate and Refine

Your first prompt won’t be perfect. Refine it!

**Trick**:  
Ask the AI to rewrite your own prompt better.

**Prompt**:

```
Here's my prompt: "Tell me how LLMs work."
Can you improve this prompt to be more specific and get better output?
```

---

## 🛠️ Bonus Prompt Engineering Tools

| Tool                                 | Use                     |
| ------------------------------------ | ----------------------- |
| [FlowGPT](https://flowgpt.com)       | Library of good prompts |
| [PromptHero](https://prompthero.com) | Image + text prompts    |
| ChatGPT Custom Instructions          | Fine-tune its behavior  |

---

## ✅ Example: Apply All Together

```
You are a startup advisor.

Task: Help me brainstorm 5 startup ideas in ed-tech.

Context: I am a college student passionate about AI and education.

Rules: Each idea should be novel, AI-based, and have a quick MVP path.
```

---

# 🎯 Final Words

Prompt engineering is NOT about "sounding smart" — it’s about **thinking clearly** and **communicating like a systems designer**.

**If you can design great inputs, you’ll unlock powerful outputs.**
