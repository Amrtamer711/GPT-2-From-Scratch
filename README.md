# GPT Implementation from Scratch

This repository contains a custom implementation of a GPT model, complete with training, evaluation, and inference capabilities. The implementation is inspired by GPT-2, with a focus on customization and educational clarity. **It is currently a work in progress and unfinished.**

# Features
- **Configurable Architecture**: Easily adjust the number of layers, embedding size, number of heads, context size, and more.
- **Custom DataLoader**: Tokenizes input data and creates batches for training.
- **From-Pretrained Suppor**t: Load weights from pre-trained Hugging Face GPT-2 models.
- **Training Pipeline**: Includes optimizer, gradient clipping, and loss tracking.
- **Supports Compilation**: Optimized with PyTorch's torch.compile for performance.
- **Inference Pipeline**: Generates text using top-k sampling.
- **Mixed Precision Training**: Optionally use torch.autocast for FP16 or BF16 precision.
- **Device Compatibility**: Automatically selects between CUDA, MPS, or CPU.

# Current Work:
- **Scheduler and Optimizer Enhancements**: A learning rate scheduler and advanced optimizer settings are not yet implemented.
- **Gradient Accumulation**: Not yet integrated for large batch sizes across GPUs.
- **Distributed Training**: DDP and multi-node setups are planned but not yet supported.
