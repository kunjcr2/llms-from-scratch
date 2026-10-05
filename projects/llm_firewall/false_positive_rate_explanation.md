# What is the Fixed 1% False Positive Rate (FPR)?

In machine learning classification—especially in security systems like this firewall—we always have to balance **security** (catching bad things) against **usability** (not interfering with legitimate usage).

## The Trade-off

1. **False Negative (FN) - Security Failure**: The firewall says a prompt is SAFE, but it's actually an INJECTION. *Result: The agent gets hijacked.*
2. **False Positive (FP) - Usability Failure**: The firewall says a prompt is an INJECTION, but it's actually SAFE (e.g., a user just asked a technical question about SQL). *Result: The user gets frustrated because their legitimate request was blocked.*

**FPR (False Positive Rate)** is the percentage of all legitimate inputs that your system accidentally blocks.

## Why fix it at exactly 1.0%?

By default, an ML model outputs a probability between `0.0` and `1.0`. The standard approach is to draw the line at `0.5`:
- `> 0.5` = Malicious
- `< 0.5` = Safe

But in production security, **usability rules everything**. If your firewall blocks 10% of legitimate user queries (FPR = 10%), users will complain, and the feature will be turned off by management immediately. 

Therefore, security engineers ask a different question: 
> *"If we configure the threshold so that **exactly 1.0%** of legitimate traffic is accidentally blocked, how many of the real attacks do we still manage to catch?"*

### Your Metric: Recall @ 1% FPR = 99.30%

What this result means conceptually for your system:

- You went into the model's probabilities and moved the threshold (to `0.9998` instead of `0.5`). 
- Because you raised the bar so high, the firewall will only accidentally block **1 in 100** normal user requests (1.0% FPR).
- Even with the bar set that high to protect usability, the firewall STILL successfully catches **99.30%** of all malicious injection attempts.

This is a **phenomenal**, production-ready result. It proves that the model isn't just randomly guessing; it has created an extreme separation between benign technical talk and actual malicious intent.
