# Neural Sort using Pointer Networks

A PyTorch implementation of a Pointer Network capable of sorting integer arrays natively. The model receives an array of integers and predicts a permutation (reordering of indices) that sorts the elements.

## What is a Pointer Network?

A [Pointer Network (Ptr-Net)](https://arxiv.org/abs/1506.03134) (Vinyals et al., 2015) is a sequence-to-sequence neural architecture designed to learn the conditional probability of an output sequence where the elements are discrete tokens corresponding to positions in an input sequence. 

While traditional sequence-to-sequence models generate a probability distribution over a fixed, pre-defined vocabulary, they struggle with problems where the output domain depends directly on the variable-length input (e.g., sorting an array of arbitrary size or the Traveling Salesperson Problem).

A Pointer Network solves this by repurposing the attention mechanism. Instead of using attention weights to blend encoder states into a context vector for a separate output layer, the Pointer Network treats the attention weights themselves as the final probability distribution. At each decoder time step, it computes attention scores over all input elements and applies a softmax function. This distribution acts as a "pointer" to select a specific element straight from the input sequence. This elegant architectural shift enables the model to natively handle tasks where the outputs are permutations or subsets of the inputs.

## Project Structure

- `data/generator.py`: Synthetic sequence generation
- `model/`: Attention-based Transformer Pointer Network
- `train.py`: Training script with curriculum learning
- `evaluate.py`: Evaluation and exact match metrics script
- `config.py`: Single source of truth for all hyperparams
- `notebook.ipynb`: Self-contained end-to-end Jupyter Notebook

## Running the Code

Install dependencies:
`pip install torch torch-vision numpy`

Train the model:
`python train.py`

Evaluate the model:
`python evaluate.py`
