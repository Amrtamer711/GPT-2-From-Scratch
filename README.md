# GPT 2 Implementation from Scratch


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

---

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

---

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

This is implemented in model.py and train.py. This was done without relying on pre-built libraries like Hugging Face's transformers.

Training
To train the model on FineWebEdu from scratch, execute:

bash
Copy code
python train.py \
  --dataset-path ./data/finewebedu/ \
  --model-size 124M \
  --batch-size 64 \
  --learning-rate 6e-4 \
  --max-steps 19000 \
  --seed 42 \
  --num-gpus 8

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
- Everything from tokenization to the training pipeline was implemented from scratch.

