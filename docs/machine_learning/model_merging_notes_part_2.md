# Model Merging Notes, Part 2

> Notes from the follow-up video covering newer MergeKit methods and Arcee Cloud usage.

## What Changed Since Part 1

This follow-up focuses on newer merge methods added after the first model merging deep dive.

- Model breadcrumbs
- Model stock
- DELLA
- A quick workflow for managed model merging in Arcee Cloud

## 1. Model Breadcrumbs

Model breadcrumbs is an improvement on task arithmetic.

- Start the same way as task arithmetic by computing task directions from fine-tuned weights minus base weights
- Then mask out a percentage of parameters instead of using every delta directly
- The mask drops both tiny and very large weights according to hyperparameters such as beta and gamma

### Intuition

The method keeps the useful middle ground while zeroing out less helpful outliers. That gives a cleaner merge signal than plain task arithmetic.

### Why it matters

- It often outperforms plain task vectors
- It scales better when merging many tasks because the merge hyperparameters are more stable
- You can tune the merge on a smaller set of tasks and then reuse those settings on many more

### Practical benefit

For large collections of fine-tuned models, breadcrumbs reduces the need for expensive hyperparameter search across every possible merge combination.

## 2. Model Stock

Model stock is a more geometric approach.

- It studies how fine-tuned tensors relate to the base model tensors
- The key idea is that fine-tuned weights with different random seeds often lie on a thin layerwise surface
- Instead of averaging many models blindly, the method estimates the center of that surface

### Intuition

If fine-tuned models all point toward a common surface in weight space, then the center of that surface is a strong candidate for a merged model.

### Why it is useful

- It can be more compute-efficient than averaging a huge number of models
- It can be done periodically during training or after training
- It often improves generalization while keeping accuracy strong on in-distribution data

### Core takeaway

Model stock is less about parameter averaging and more about finding a good geometric center for a family of fine-tuned solutions.

## 3. DELLA

DELLA stands for Drop and Rescale via Sampling with Magnitude.

- Compute task deltas as usual
- Drop parameters probabilistically, with small-magnitude values more likely to be removed
- Use a sign-election step similar to TIES
- Fuse the surviving values into the merged update

### What is different from TIES

DELLA uses stochastic dropping based on magnitude rather than a more deterministic trimming rule. The sampling makes the pruning step probabilistic instead of hard-thresholded.

### Why it works well

- It tends to outperform plain task arithmetic
- It scales well to many tasks
- It often improves the base model on individual tasks after merging other skills into it

### Main idea

DELLA is still a merge method, but it first compresses and sparsifies the task updates so the surviving signal is stronger and easier to merge.

## 4. Arcee Cloud Merge Workflow

The video also shows a managed merge workflow in Arcee Cloud.

- Sign up for Arcee Cloud
- Open the merging tab
- Create a merge job
- Provide a YAML config file for MergeKit
- Launch the merge

### Why use it

- Free tier available for merges
- No need to run the merge locally on your own machine
- Convenient if you want a managed interface around MergeKit

## Practical Summary

- Breadcrumbs is task arithmetic with masking and better scaling behavior
- Model stock uses geometry to find a center of fine-tuned solutions
- DELLA combines probabilistic dropping with sign selection and merging
- Arcee Cloud provides a simple hosted path for running MergeKit merges

## Overall Takeaway

The newer merge methods focus less on simple averaging and more on filtering, geometry, and stability at scale. That makes them better suited to large collections of task-specific models and broader multitask merge workflows.