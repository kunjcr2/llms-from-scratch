# PyTorch Mastery: A Senior Engineer's Reference

This folder is a comprehensive, production-oriented reference for PyTorch. It is designed for Machine Learning, Deep Learning, NLP, and Computer Vision engineers who need to quickly recall syntax, best practices, and implementation patterns. This is not a tutorial; it is a scannable knowledge base where every notebook is immediately runnable and reflects industry-standard practices as of PyTorch 2.x.

## Master Index

| Notebook | Core Coverage | When to Use It |
| :--- | :--- | :--- |
| **01 Core PyTorch** | | |
| [01 Tensors](01_core_pytorch/01_tensors.ipynb) | Ops, dtypes, shapes, broadcasting | Foundation for all PyTorch work |
| [02 Autograd](01_core_pytorch/02_autograd.ipynb) | Gradients, computational graphs, hooks | Debugging custom loss/backprop |
| [03 NN Module](01_core_pytorch/03_nn_module.ipynb) | Layers, custom blocks, state_dict | Building model architectures |
| [04 Optimizers & Loss](01_core_pytorch/04_optimizers_loss.ipynb) | SGD, AdamW, CrossEntropy, Custom | Configuring training objectives |
| [05 Data Loading](01_core_pytorch/05_data_loading.ipynb) | Dataset, DataLoader, Samplers | Handling I/O bottlenecks |
| [06 GPU & Performance](01_core_pytorch/06_gpu_cuda_performance.ipynb) | CUDA, Mixed Precision, `torch.compile` | Scaling and speed optimization |
| **02 Deep Learning** | | |
| [01 Training Loop](02_deep_learning/01_training_loop.ipynb) | Boilerplate, checkpoints, validation | Implementing standard training workflows |
| [02 CNNs](02_deep_learning/02_cnns.ipynb) | Conv2d, Pooling, BatchNorm, Residuals | Image and spatial data processing |
| [03 RNNs & LSTMs](02_deep_learning/03_rnns_lstms.ipynb) | Sequential modeling, hidden states | Time-series and legacy NLP |
| [04 Regularization](02_deep_learning/04_regularization_tricks.ipynb) | Dropout, Weight Decay, Augmentation | Combating overfitting |
| [PROJECT Image Classifier](02_deep_learning/PROJECT_image_classifier.ipynb) | Custom CNN on CIFAR-10 | End-to-end vision baseline |
| **03 Computer Vision** | | |
| [01 Transforms](03_computer_vision/01_torchvision_transforms.ipynb) | Augmentation, v2 transforms | Preprocessing image data |
| [02 Finetuning](03_computer_vision/02_pretrained_model_fine_tuning.ipynb) | Transfer learning, head replacement | Adapting SOTA models to new tasks |
| [03 Object Detection](03_computer_vision/03_object_detection.ipynb) | FasterRCNN, YOLO patterns, Bboxes | Detecting and localizing objects |
| [04 ViT & CLIP](03_computer_vision/04_clip_and_vision_transformers.ipynb) | Vision Transformers, Multimodal | Modern SOTA vision approaches |
| [PROJECT Finetune ResNet](03_computer_vision/project_fine_tune_resnet.ipynb) | ResNet-50 discriminative LRs | Real-world transfer learning |
| **04 NLP & Transformers** | | |
| [01 Tokenization](04_nlp_transformers_llms/01_tokenization_embeddings.ipynb) | BPE, WordPiece, Embedding layers | Text-to-vector foundations |
| [02 Attention](04_nlp_transformers_llms/02_attention_mechanism.ipynb) | Self-attention, Multi-head attention | Deep dive into Transformer core |
| [03 Transformer Architecture](04_nlp_transformers_llms/03_transformer_architecture.ipynb) | Encoder, Decoder, LayerNorm | Understanding GPT/BERT internals |
| [04 HF Pipelines](04_nlp_transformers_llms/04_huggingface_pipeline.ipynb) | Quick inference, task-specific heads | Rapid prototyping with Transformers |
| [05 Finetuning LLMs](04_nlp_transformers_llms/05_fine_tuning_llms.ipynb) | LoRA, QLoRA, PEFT basics | Adapting large models efficiently |
| [06 RAG Basics](04_nlp_transformers_llms/06_rag_basics.ipynb) | Vector DBs, retrieval contexts | Building augmented generation apps |
| [PROJECT BERT Classifier](04_nlp_transformers_llms/project_fine_tune_bert_classifier.ipynb) | BERT on SST-2/IMDb | Industry standard NLP classification |
| **05 Classical ML** | | |
| [01 Sklearn Essentials](05_ml_classical/01_sklearn_essentials.ipynb) | Pipelines, preprocessing, metrics | Classical ML basics for DL engineers |
| [02 Tabular PyTorch](05_ml_classical/02_pytorch_for_tabular.ipynb) | Entity embeddings, MLP for tables | DL approach to structured data |
| [PROJECT Tabular Classifier](05_ml_classical/PROJECT_tabular_classifier.ipynb) | Titanic/Adult PyTorch vs XGBoost | Comparing NN vs GBDT performance |

## How to Navigate This Reference

1.  **Search First**: Every notebook follows a strict header structure. Use `Ctrl+F` for keywords like `✅ Use when` or `❌ Don't use when`.
2.  **Runnable Code**: Every code block is self-contained. You can copy-paste any cell into your production environment; imports and device setup are consistent throughout.
3.  **Complexity Scaling**: Small utilities have 1-liners; complex APIs have parameter tables and pitfall sections.

## Prerequisites

Ensure you have a Python 3.10+ environment with a GPU (preferred) or CPU.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tokenizers timm scikit-learn pandas matplotlib seaborn
```

## Global Standards

-   **PyTorch 2.x**: Heavy emphasis on `torch.compile`.
-   **Hardware Agnostic**: Every notebook uses `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
-   **Modern NLP**: HuggingFace is treated as the default for all NLP tasks.
-   **Modern CV**: `torchvision` and `timm` are standard.
