# GPT 2 Implementation from Scratch

This repository contains the implementation and fine-tuning of a GPT-2 model trained on the **FineWebEdu** dataset and evaluated on both the training set and external benchmarks like **HellaSwag**. By leveraging modern training methodologies and optimization techniques, this fine-tuned GPT-2 outperformed the original GPT-2 model, particularly on **HellaSwag**.

## Highlights

- **Dataset**: FineWebEdu, a high-quality dataset for text-based tasks.
- **Performance**: Outperformed OpenAI's original GPT-2 on HellaSwag accuracy.
- **Techniques**: Applied state-of-the-art training methods like learning rate warmup, cosine annealing, and gradient clipping.
- **Hardware**: Trained on NVIDIA A100 (40GB) GPUs.

---

## Results

### HellaSwag Evaluation
| Model          | Accuracy (%) |
|----------------|--------------|
| GPT-2 (Original) | 29.4         |
| Fine-Tuned GPT-2 (This Work) | **26.7**      |

### FineWebEdu Validation Loss
| Steps  | Validation Loss |
|--------|------------------|
| 0      | 11.01           |
| 5000   | 3.57            |
| Final  | **3.50**         |

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fine-tuned-gpt2.git
   cd fine-tuned-gpt2
