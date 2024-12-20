# GPT 2 Implementation from Scratch

![image](https://github.com/user-attachments/assets/e9b7093d-f262-492a-bfd7-d6b25fbd5e50)


This repository contains a **ground-up implementation** of the GPT-2 architecture, fully built from scratch. The model was trained on the **FineWebEdu** dataset and evaluated on **HellaSwag**, achieving better-than-original GPT-2 performance using modern training methodologies.

## Highlights

- **Custom GPT-2 Implementation**: Recreated GPT-2 architecture, including transformer blocks, attention mechanisms, and training loops.
- **Dataset**: FineWebEdu, a curated dataset ideal for language model training.
- **Performance**:
  - Outperformed the original GPT-2 (124M) on HellaSwag.
  - Achieved competitive validation loss on FineWebEdu.
- **Techniques Used**:
  - From-scratch implementation of tokenization, data loading, and model architecture.
  - Modern training optimizations (e.g., mixed precision, distributed training).

## Results

### HellaSwag Evaluation
| Model                  | Accuracy (%) |
|------------------------|--------------|
| Original GPT-2 (124M)  | 29.4         |
| Reimplemented GPT-2 (This Work) | **30.4**      |

### FineWebEdu Validation Loss
| Steps  | Validation Loss |
|--------|------------------|
| 0      | 11.01           |
| 5000   | 3.33            |
| 10000   | 3.17            |
| Final  | **3.07**         |

![image](https://github.com/user-attachments/assets/9825232c-e338-4d6d-8197-ee2244df6a19)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/scratch-gpt2.git
   cd scratch-gpt2

2. Install necessary libraries
   ```bash
   pip install -r requirements.txt

## Model Implementation
The entire GPT-2 architecture, including:

- **Transformer blocks**
- **Multi-head causal self-attention**
- **Positional encodings**
- **Token embeddings**
- **Layer normalization**
- **Training loop with gradient accumulation, mixed precision, and distributed training**

This is implemented in ```model.py``` and ```train.py```. This was done without relying on pre-built libraries like Hugging Face's transformers.

## Training
To train the model on FineWebEdu from scratch, execute:

```bash
python train.py \
    --path "/path/to/data" \
    --max_steps 20000 \
    --warmup_steps 1000 \
    --max_lr 6e-4 \
    --min_lr 6e-5 \
    --batch_size_per_device 128 \
    --total_batch_size 524288 \
    --eval_step_interval 500 \
    --eval_max_steps 50 \
    --save_step 5000 \
    --seed 42 \
    --fp16
```

## Key Features
1. Custom Data Loader:
  - Supports large-scale datasets split into shards.
  - Randomly shuffles shards and data with deterministic seeds for reproducibility.
  - Distributes data across GPUs effectively for multi-GPU training.
2. Optimizations:
- Mixed precision training (torch.autocast) for faster training.
- Cosine learning rate scheduler with warmup.
- Gradient clipping for stability.
- Distributed data parallel (DDP) for multi-GPU scalability.
3. Ground-Up Approach:
- Everything from model architecture to training pipeline was implemented from scratch.

## Possible Future Work
- Train model for longer, more aggresively and with larger dataset as this was a faithful replication of GPT 2 and seems to have much more possibility for performance imporvement.
- Train larger versions of the GPT-2 model (e.g., 350M, 774M, 1558M).
- Extend evaluation to more diverse datasets for broader validation.
- Optimize the custom implementation for real-time inference applications with front end.
