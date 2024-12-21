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
| My GPT-2 (124M) | **30.4**      |

### FineWebEdu Validation Loss
| Steps  | Validation Loss |
|--------|------------------|
| 0      | 11.01           |
| 5000   | 3.33            |
| 10000   | 3.17            |
| Final  | **3.07**         |

![image](https://github.com/user-attachments/assets/9825232c-e338-4d6d-8197-ee2244df6a19)

## Model Implementation
The entire GPT-2 architecture, including:

- **Transformer blocks**
- **Multi-head causal self-attention**
- **Positional encodings**
- **Token embeddings**
- **Layer normalization**
- **Training loop with gradient accumulation, mixed precision, and distributed training**

This is implemented in ```model.py``` and ```train.py```. This was done without relying on pre-built libraries like Hugging Face's transformers.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amrtamer711/gpt-2-from-scratch.git
   cd gpt-2-from-scratch

2. Install necessary libraries
   ```bash
   pip install -r requirements.txt
   
## Dataset Preparation

To prepare the FineWebEdu Dataset, execute:
  ```bash
  python fineweb.py
```

### Notes:
- This may take a while to execute as this dependent on the number of cores on your machine.
- It will produce 100 shards of the dataset and will require approx. 20 GB of memory to be stored.

## Training
To train the model on FineWebEdu from scratch, execute:

```bash
torchrun --standalone --nproc_per_node=x train.py \
    --path "/path/to/data" \
    --max_steps 19073 \
    --warmup_steps 715 \
    --max_lr 6e-4 \
    --min_lr 6e-5 \
    --batch_size_per_device 64 \
    --total_batch_size 524288 \
    --eval_step_interval 250 \
    --eval_max_steps 20 \
    --save_step 5000 \
    --seed 1337 \
    --fp16
```
where x in ```--nproc_per_node=x``` should be the number of GPUs you have available. 

## Results Analysis
To evaluate the model and compare it to the original versions of GPT 2 and GPT 3, run the ```benchmark.ipynb``` notebook. It will extract the latest log in the log file.

### Notes:
- These were the exact parameters I used to produce these results, in order to replicate the conditions of GPT 2, but it seems that slightly more aggressive training would produce better results.
- Modify the batch size per device until maximum untilization of your GPUs.

## Model Generation
To generate from the model, run  ```generate.ipynb``` where you can modify the input text.

## Key Features
1. Custom Data Loader:
   - Supports large-scale datasets split into shards.
   - Randomly shuffles shards and data with deterministic seeds for reproducibility.
   - Distributes data across GPUs effectively for multi-GPU training.
2. Optimizations:
   - Optional Fused AdamW and Flash Attention for more efficient training.
   - Mixed precision training for faster training.
   - Cosine learning rate scheduler with warmup.
   - Gradient clipping for stability.
   - Distributed data parallel (DDP) for multi-GPU scalability.
3. Ground-Up Approach:
   - Everything from model architecture to training pipeline was implemented from scratch.

## Possible Future Work
- Train model for longer, more aggresively and with larger dataset as this was a faithful replication of GPT 2 and seems to have much more possibility for performance imporvement.
- Fine-tune it for user-model chat capabilities. 
- Train larger versions of the GPT-2 model (e.g., 350M, 774M, 1558M).
- Extend evaluation to more diverse datasets for broader validation.
- Optimize the custom implementation for real-time inference applications with front end.
