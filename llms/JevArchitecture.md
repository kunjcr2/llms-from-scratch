# Jev Architecture - Student Notes

Source: [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/?v=3) by Archer Hume (17 September 2026).

> Important: Jev is closed-source. These notes describe the article's **best reconstruction** from API experiments, not a confirmed internal design. The strongest claims are that it returns decision probabilities directly and can handle many questions in parallel. Details such as the exact transformer layout and use of MoE are informed guesses.

## The main idea

Normal chat-style LLMs are built to write the next token, then the next, then the next. If we ask one for a support-ticket route, it may generate JSON such as:

```json
{ "route": "payments", "confidence": 0.91 }
```

Jev appears to target a narrower job: make a structured decision from a fixed set of allowed answers. Instead of spending model steps spelling out `payments` and `0.91`, it reads the input and returns the probabilities for the choices directly.

For example, given a complaint about failed payouts, it could return:

```text
payments: 0.91    account: 0.03    other: 0.06
urgent:   0.42    not urgent: 0.58
```

The calling program can turn those numbers into JSON and apply its own rules. This is useful for routing, moderation, fraud checks, and risk decisions, where software needs a choice and an uncertainty estimate, not a paragraph of prose.

## Proposed computation flow

```text
shared state / evidence
        |
        |  encode once
        v
shared internal representation (and cache)
        |
        +--> question 1 + its options --> probabilities for question 1
        +--> question 2 + its options --> probabilities for question 2
        +--> question 3 + its options --> probabilities for question 3
```

The key design choice is **shared evidence, separate questions**:

- The long common input (for example, a customer message or incident report) is processed once.
- Each question receives that shared context plus its own instructions and answer choices.
- Question branches are meant to be independent: a question about urgency should not alter a separate question about which team receives the ticket.
- Since the questions do not have to wait for each other's answers, the server can process them together in a batch.

This is much cheaper than sending the same long report to a regular LLM once per question. It does **not** remove every cost: each question still has to examine the shared evidence. It mainly removes repeated work for the common input.

## How it differs from normal LLM generation

### Ordinary autoregressive LLM

1. Read the prompt.
2. Predict one token.
3. Feed that token back in.
4. Repeat until the answer is complete.

This loop is necessary when producing free-form text, because the next token depends on the previous generated token.

### Proposed Jev-style decision model

1. Read the state/evidence.
2. Read a question and all its allowed options.
3. Produce scores for the options and convert them into probabilities.
4. Stop.

There is no need to generate a sentence token by token. In simple terms, the model ends with a small decision layer (a *readout*) instead of a text-generation loop.

For a multiple-choice question, the readout gives every option a score and normalizes the scores so they add to 1. A yes/no question can be treated similarly with two possible outcomes.

## Why the answer choices matter together

The article's experiments suggest Jev does not score every option completely independently. It seems to read the whole option list before deciding.

That makes sense. The meaning of an answer can depend on the alternatives:

- `none of the above` is impossible to understand without the other options.
- `other` means something different depending on what categories were already listed.
- Two very similar options require comparison, not isolated scoring.

One experiment added an irrelevant extra option and found that the relative probabilities of two existing options changed. If each option had a fixed independent score, adding another option should mostly just divide the same probability mass differently. The shift suggests that the full list affects the decision.

### Practical caution: option order can matter

The article also found that reordering otherwise identical options could change the returned probabilities. This does not mean the model is useless, but it is a warning for real systems:

- Keep option wording and order consistent when possible.
- Test several orderings before deploying a threshold-based rule.
- Do not assume that a returned probability is perfectly stable just because the labels are the same.

## Probability versus "confidence"

The central claim is not simply "Jev gives confidence scores." A normal LLM can write `90% confident`, but that text is not automatically a reliable 90% chance of being correct.

The intended goal here is **calibration**: over many cases where the model assigns about 0.8 probability, it should be correct about 80% of the time.

The company calls its post-training approach RLCD (Reinforcement Learning for Calibrated Decisions). The exact training recipe is not public, but the general idea is sensible: train the model using real outcomes so that its probability distribution is useful for decisions, not just fluent-sounding.

Calibration is measured across many examples, never from one answer alone. A model can be very confident and still be wrong. Also, calibration can get worse when the real-world data changes from the data used for training or evaluation.

The API's separate field named `confidence` should not be confused with a second learned prediction. According to the article, it is calculated from the returned option distribution - roughly, how much the leading choice stands above an even split. A concentrated distribution can still be wrong.

## Likely underlying model

The author thinks the base is probably a **causal transformer**: the same general family as most modern text-generating LLMs. Its normal prompt-processing stage is reused, but it stops before next-token generation.

The article also suspects a **Mixture of Experts (MoE)** backbone. In an MoE model, only a few specialist sub-networks are activated for a token, allowing a large model to run with less computation than a fully dense model. This could help explain very fast processing of long inputs, but it is the least certain part of the reconstruction. The proposed decision interface would still work with an ordinary dense transformer.

## What the experiments support - and what they do not

| Reasonably supported | Still uncertain / inferred |
| --- | --- |
| Direct distributions over a typed set of answers | The exact readout implementation |
| Shared state can be reused across questions | The exact attention mask and cache layout |
| Questions appear behaviourally isolated | The identity of the pretrained base model |
| Options influence a joint decision | Whether the backbone uses MoE |
| Many independent question branches can be batched | Hardware, number of workers, and numeric precision |

This distinction matters. Observing an API's behaviour can rule out some designs, but it rarely proves one unique internal architecture.

## Limitation of parallel questions

Parallel branches work only when questions are independent given the shared state.

For example, "Which team should receive this ticket?" and "Is it urgent?" can be answered side by side. But if question B truly needs the result of question A, the application must run another stage, or combine them into one larger decision. Parallelism cannot remove a real logical dependency.

## Takeaway

Jev is best understood as a general LLM repurposed as a **decision engine**:

1. Read shared evidence once.
2. Evaluate several independent typed questions in parallel.
3. Compare all choices for each question.
4. Return numerical probability distributions directly.
5. Let normal software format the result and make the final policy decision.

The novel part is not that classification exists - classifiers have existed for a long time. It is combining an LLM's broad understanding with shared computation, a structured decision interface, and an attempt to make the output probabilities reliable enough for software to use.
