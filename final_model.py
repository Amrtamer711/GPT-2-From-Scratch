# Required Libraries
from contextlib import nullcontext  # For conditional context management
import torch  # PyTorch library for building and training models
import torch.nn as nn  # Neural network module in PyTorch
import torch.nn.functional as F  # Functional interface for neural network operations
from dataclasses import dataclass  # To create simple data structures
import math  # For mathematical computations
import tiktoken  # For tokenizing input text
import time  # To measure execution time
import inspect  # To inspect function signatures
import os  # For file and environment variable operations
import torch.distributed as dist  # For distributed training
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP module for multi-GPU training
from hellaswag import render_example, iterate_examples  # Specific dataset functions
import numpy as np

# Path for input data and saving the model
path = ""  # Directory for saving/loading data and models

# Configuration for the GPT model
@dataclass
class GPTConfig:
    context_size: int = 1024  # Sequence length
    vocab_size: int = 50304  # Vocabulary size for tokenization (GPT-2 uses 50257)
    num_blocks: int = 12  # Number of transformer blocks
    num_heads: int = 12  # Number of attention heads
    embed_size: int = 768  # Size of embeddings
    dropout: float = 0.0  # Dropout rate for regularization
    bias: bool = True  # Use biases in linear layers

# Check if Distributed Data Parallel (DDP) is enabled
ddp = int(os.environ.get('RANK', -1)) != -1  # Check if running in a distributed environment
if ddp:
    # Initialize DDP settings
    dist.init_process_group(backend='nccl')  # Use NCCL backend for distributed training
    rank = int(os.environ['RANK'])  # Global rank of the process
    local_rank = int(os.environ['LOCAL_RANK'])  # Local rank of the process
    world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
    device = f'cuda:{local_rank}'  # Assign the local GPU
    device_type = 'cuda'
    torch.cuda.set_device(device)  # Set the device for this process
    is_master = (local_rank == 0)  # Identify the master process for logging
    if is_master:
        print("Distributed learning ON")
    print(f"Global Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
else:
    # Fallback for single-device training
    rank, local_rank, world_size = 0, 0, 1  # Default ranks and world size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, else CPU
    device_type = device
    is_master = True  # Single process acts as the master

    # Log the device being used
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)} (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use Metal Performance Shaders on macOS
        print("Using device: MPS")
    else:
        print("Using device: CPU")

# Global training configurations
compiled = False  # Use PyTorch's Compiler
fp16 = True  # Enable mixed-precision training for faster computation
dtype = 'TF32'  # Default floating-point precision for matrix multiplication
total_batch_size = 524288  # Total number of tokens in a batch
batch_size_per_device = 64  # Number of sequences processed per GPU
seed = 1337  # Seed for reproducibility
eval_step_interval = 250  # Evaluate every 250 steps
eval_max_steps = 20  # Number of steps for evaluation
save_step = 5000  # Save model every 5000 steps

max_steps = 19073  # Total number of training steps
warmup_steps = 715  # Warmup steps for learning rate scheduler

max_lr = 6e-4  # Maximum learning rate
min_lr = max_lr * 0.1  # Minimum learning rate after decay

# Calculate gradient accumulation steps
grad_accum_steps = total_batch_size // (batch_size_per_device * GPTConfig().context_size * world_size)
if is_master:
    print("The desired amount of tokens in a single batch is:", total_batch_size)
    print("Gradient Accumulation Steps:", grad_accum_steps)

# GPT Model Components and Classes
class CasualSelfAttention(nn.Module):
    """
    Implements scaled dot-product attention with optional dropout and causal masking.
    """
    def __init__(self, config):
        super().__init__()
        # Projections for Query, Key, and Value
        self.c_attn = nn.Linear(config.embed_size, 3 * config.embed_size)
        self.c_proj = nn.Linear(config.embed_size, config.embed_size)  # Output projection
        self.num_heads = config.num_heads  # Number of attention heads
        self.embed_size = config.embed_size  # Embedding size
        self.dropout = config.dropout  # Dropout probability
        self.att_dropout = nn.Dropout(config.dropout)  # Dropout for attention scores
        self.residual_dropout = nn.Dropout(config.dropout)  # Dropout for residual connections
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # Check if Flash Attention is available
        if not self.flash:
            if is_master:
                print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask for slower attention implementation
            self.register_buffer("bias", torch.tril(torch.ones(config.context_size, config.context_size))
                                  .view(1, 1, config.context_size, config.context_size))
    
    def forward(self, x):
        # Input shape: (Batch size, Sequence length, Embedding size)
        B, T, C = x.size()  # Extract batch size (B), sequence length (T), and embedding size (C)

        # Compute Query, Key, Value projections
        qkv = self.c_attn(x)  # Linear projection to generate Q, K, V
        q, k, v = qkv.split(self.embed_size, dim=2)  # Split into query, key, and value

        # Reshape Q, K, V for multi-head attention
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, Heads, T, Head Size)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        if self.flash:  # Fast attention (requires PyTorch >= 2.0)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=self.dropout if self.training else 0)
        else:  # Slow attention (uses explicit causal mask)
            # Scaled dot-product attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # Scale by sqrt(head size)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # Apply causal mask
            att = F.softmax(att, dim=-1)  # Softmax over the last dimension
            if self.training:
                att = self.att_dropout(att)  # Apply dropout to attention scores
            y = att @ v  # Compute weighted sum of values

        # Combine heads and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Reshape back to (B, T, C)
        y = self.c_proj(y)  # Final linear projection
        if self.training:
            y = self.residual_dropout(y)  # Apply dropout to output
        return y  # Return attention output

class MLP(nn.Module):
    """
    Multi-layer perceptron used in transformer blocks.
    """
    def __init__(self, config):
        super().__init__()
        # Two fully connected layers with a GeLU activation in between
        self.c_fc = nn.Linear(config.embed_size, 4 * config.embed_size, bias=config.bias)  # Expand dimension
        self.gelu = nn.GELU(approximate='tanh')  # GeLU activation function
        self.c_proj = nn.Linear(4 * config.embed_size, config.embed_size, bias=config.bias)  # Reduce dimension
        self.dropout = nn.Dropout(config.dropout)  # Dropout for regularization

    def forward(self, x):
        # Feedforward pass
        x = self.c_fc(x)  # First fully connected layer
        x = self.gelu(x)  # GeLU activation
        x = self.c_proj(x)  # Second fully connected layer
        if self.training:
            x = self.dropout(x)  # Apply dropout during training
        return x  # Return processed output

class Block(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_size, bias=config.bias)  # Layer normalization before attention
        self.ln_2 = nn.LayerNorm(config.embed_size, bias=config.bias)  # Layer normalization before MLP
        self.attn = CasualSelfAttention(config)  # Causal self-attention module
        self.mlp = MLP(config)  # Multi-layer perceptron (feed-forward network)

    def forward(self, x):
        # Apply attention with residual connection
        x = x + self.attn(self.ln_1(x))  # LayerNorm -> Attention -> Residual
        # Apply feed-forward network with residual connection
        x = x + self.mlp(self.ln_2(x))  # LayerNorm -> MLP -> Residual
        return x  # Return processed output

class GPT(nn.Module):
    """
    GPT-like model implementation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config  # Store configuration

        # Transformer components
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embed_size),  # Token embeddings
            wpe=nn.Embedding(config.context_size, config.embed_size),  # Positional embeddings
            dropout=nn.Dropout(config.dropout),  # Dropout layer
            h=nn.ModuleList([Block(config) for _ in range(config.num_blocks)]),  # Stack of transformer blocks
            ln_f=nn.LayerNorm(config.embed_size, bias=config.bias)  # Final layer normalization
        ))

        self.lm_head = nn.Linear(config.embed_size, config.vocab_size, bias=False)  # Output layer for logits

        # Tie input embedding weights with the output layer
        self.transformer.wte.weight = self.lm_head.weight  # Weight tying

        # Initialize weights
        self.apply(self.init_weights)  # Apply weight initialization
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):  # Special initialization for projection layers
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_blocks))

    def init_weights(self, module):
        """
        Initialize weights for the model layers.
        """
        std = 0.02  # Standard deviation for initialization
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)  # Initialize weights with Gaussian
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # Initialize biases to zero
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)  # Initialize embeddings with Gaussian

    def forward(self, x, targets=None):
        """
        Forward pass for training and inference.
        """
        B, T = x.size()  # Batch size and sequence length

        # Combine token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)  # Positional indices
        pos_emb = self.transformer.wpe(pos)  # Positional embeddings
        token_emb = self.transformer.wte(x)  # Token embeddings
        x = pos_emb + token_emb  # Combine token and positional embeddings
        if self.training:
            x = self.transformer.dropout(x)  # Apply dropout during training

        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)  # Final layer normalization

        # Compute loss for training or logits for inference
        logits = self.lm_head(x)  # Compute logits for all tokens

        loss = None
        if targets is not None:  # Training mode
            # logits = self.lm_head(x)  # Compute logits for all tokens
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # Compute cross-entropy loss
        # else:  # Inference mode
        #     logits = self.lm_head(x[:, [-1], :])  # Predict next token logits (last token in the sequence)

        return logits, loss  # Return logits and loss (if applicable)

    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens given input tokens.
        """
        with torch.no_grad():  # Disable gradient computation
            for i in range(max_new_tokens):
                # Truncate sequence to context size if necessary
                tokens_cropped = tokens if tokens.size(1) <= self.config.context_size else tokens[:, -self.config.context_size:]
                logits, _ = self(tokens_cropped)  # Forward pass
                logits = logits[:, -1, :] / temperature  # Scale logits by temperature

                if top_k is not None:  # Apply top-k sampling
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # Top-k probabilities
                    logits[logits < v[:, [-1]]] = -float('Inf')  # Mask logits below the top-k threshold

                probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
                new_token = torch.multinomial(probs, num_samples=1)  # Sample next token
                tokens = torch.cat((tokens, new_token), dim=1)  # Append the new token to the sequence

        return tokens  # Return the generated tokens
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, device_type):
        """
        Configure the optimizer with different parameter groups.
        """
        # Collect all parameters and filter those that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate parameters into those with weight decay and those without
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # Weights in matrices (2D or higher)
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # Biases and normalization weights

        # Create optimizer parameter groups
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},  # Apply weight decay
            {'params': nodecay_params, 'weight_decay': 0.0}  # No weight decay
        ]

        # Log the number of parameters in each group
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if is_master:
            print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Check if fused AdamW is available for better performance
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        # Create the AdamW optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args)
        if is_master:
            print(f"Using fused AdamW: {use_fused}")

        return optimizer  # Return the configured optimizer

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        Load a pretrained GPT model and initialize it with specified configuration.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}  # Ensure valid model type
        override_args = override_args or {}  # Use provided overrides or default to an empty dict

        # Validate that only the dropout rate can be overridden
        assert all(k == 'dropout' for k in override_args)

        from transformers import GPT2LMHeadModel  # Import Hugging Face's GPT2 model

        print("Loading weights from pretrained GPT: %s" % model_type)

        # Model configurations based on model type
        config_args = {
            'gpt2': dict(num_blocks=12, num_heads=12, embed_size=768),  # 124M parameters
            'gpt2-medium': dict(num_blocks=24, num_heads=16, embed_size=1024),  # 350M parameters
            'gpt2-large': dict(num_blocks=36, num_heads=20, embed_size=1280),  # 774M parameters
            'gpt2-xl': dict(num_blocks=48, num_heads=25, embed_size=1600)  # 1558M parameters
        }[model_type]

        # Enforce default configurations
        print("Forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # Fixed vocabulary size for GPT models
        config_args['context_size'] = 1024  # Fixed sequence length
        config_args['bias'] = True  # Always use bias

        # Override dropout if specified
        if 'dropout' in override_args:
            print(f"Overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # Initialize a new GPT model with the specified configuration
        config = GPTConfig(**config_args)
        model = GPT(config)

        # Load pretrained weights from Hugging Face
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Map weights from Hugging Face model to custom GPT implementation
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']  # Layers to transpose
        for k in model.state_dict().keys():
            if k in transposed:
                with torch.no_grad():
                    model.state_dict()[k].copy_(sd_hf[k].t())  # Transpose weights
            else:
                with torch.no_grad():
                    model.state_dict()[k].copy_(sd_hf[k])  # Directly copy weights

        return model  # Return the initialized GPT model

# Learning rate scheduler using a cosine decay with warmup
def get_lr(step):
    if step < warmup_steps:
        return ((step + 1) / warmup_steps) * max_lr  # Linear warmup
    if step > max_steps:
        return min_lr  # Return minimum LR after decay

    # Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine function
    return min_lr + coeff * (max_lr - min_lr)  # Adjust learning rate based on cosine decay

# Function to compute the most likely completion from logits
def get_most_likely_row(tokens, mask, logits):
    """
    Computes the most likely completion row given the tokens and logits.
    """
    # Shift logits and tokens for autoregressive loss
    shift_logits = logits[..., :-1, :].contiguous()  # Remove the last token
    shift_tokens = tokens[..., 1:].contiguous()  # Shift targets to align with logits

    # Compute loss for each position
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # Flatten logits
    flat_shift_tokens = shift_tokens.view(-1)  # Flatten tokens
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')  # Compute per-token loss
    shift_losses = shift_losses.view(tokens.size(0), -1)  # Reshape to batch size

    # Mask the losses for completion region
    shift_mask = mask[..., 1:].contiguous()  # Shift mask to match shifted tokens
    masked_shift_losses = shift_losses * shift_mask  # Apply mask
    avg_loss = masked_shift_losses.sum(dim=1) / shift_mask.sum(dim=1)  # Average loss per row

    # Return the index of the row with the lowest loss
    return avg_loss.argmin().item()

# Function to load tokens from a file
def load_tokens(filename):
    """
    Loads and converts tokenized data from a file into a PyTorch tensor.
    """
    npt = np.load(filename)  # Load NumPy array
    npt = npt.astype(np.int32)  # Convert to int32
    return torch.tensor(npt, dtype=torch.long)  # Convert to PyTorch tensor

class DataLoader:
    """
    A custom data loader for loading tokenized data in shards.
    """
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B  # Batch size
        self.T = T  # Sequence length
        self.process_rank = process_rank  # Process rank for distributed loading
        self.num_processes = num_processes  # Total number of processes
        assert split in {'train', 'val'}  # Ensure valid split

        # Locate shards for the specified split
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]  # Filter shards for the given split
        shards = sorted(shards)  # Sort shard filenames
        shards = [os.path.join(data_root, s) for s in shards]  # Get full paths
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"  # Ensure shards exist
        if is_master:
            print(f"Found {len(shards)} shards for split {split}")
        self.reset()  # Initialize shard loading

    def reset(self):
        """
        Resets the state of the data loader, starting at the first shard.
        """
        self.current_shard = 0  # Reset to the first shard
        self.tokens = load_tokens(self.shards[self.current_shard])  # Load first shard
        self.current_position = self.B * self.T * self.process_rank  # Calculate starting position

    def next_batch(self):
        """
        Fetches the next batch of tokenized data.
        """
        # Slice tokens to create input (x) and target (y) sequences
        buf = self.tokens[self.current_position: self.current_position + self.B * self.T + 1]
        x = buf[:-1].view(self.B, self.T)  # Inputs (excluding last token)
        y = buf[1:].view(self.B, self.T)  # Targets (shifted by one)

        # Advance position for the next batch
        self.current_position += self.B * self.T * self.num_processes

        # Check if we need to load the next shard
        if self.current_position + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)  # Move to the next shard
            self.tokens = load_tokens(self.shards[self.current_shard])  # Load the new shard
            self.current_position = self.B * self.T * self.process_rank  # Reset position
        return x, y  # Return input and target tensors

# class DataLoader:
#     """
#     A custom data loader for loading tokenized data in shards.
#     """
#     def __init__(self, B, T, process_rank, num_processes, split, shuffle=True):
#         self.B = B  # Batch size
#         self.T = T  # Sequence length
#         self.process_rank = process_rank  # Process rank for distributed loading
#         self.num_processes = num_processes  # Total number of processes
#         self.shuffle = shuffle  # Whether to shuffle the data
#         assert split in {'train', 'val'}, f"Invalid split: {split}"  # Ensure valid split

#         # Locate shards for the specified split
#         data_root = "edu_fineweb10B"
#         shards = os.listdir(data_root)
#         shards = [s for s in shards if split in s]  # Filter shards for the given split
#         shards = sorted(shards)  # Sort shard filenames
#         shards = [os.path.join(data_root, s) for s in shards]  # Get full paths
#         self.shards = shards
#         assert len(shards) > 0, f"No shards found for split {split}"  # Ensure shards exist
#         if is_master:
#             print(f"Found {len(shards)} shards for split {split}")
#         self.reset()  # Initialize shard loading

#     def reset(self):
#         """
#         Resets the state of the data loader, ensuring all shards are utilized.
#         Optionally shuffles the shards and tokens within each shard.
#         """
#         if self.shuffle:
#             random.seed(42 + self.process_rank)  # Seed for reproducibility
#             random.shuffle(self.shards)  # Shuffle the order of shards

#         # Load the first shard and shuffle its tokens if needed
#         self.current_shard = 0  # Start from the first shard
#         self.tokens = self._load_and_shuffle_tokens(self.shards[self.current_shard])  # Load and shuffle tokens
#         self.current_position = self.B * self.T * self.process_rank  # Calculate starting position

#     def _load_and_shuffle_tokens(self, shard):
#         """
#         Loads tokens from a shard and shuffles them if required.
#         """
#         tokens = load_tokens(shard)  # Load tokens from the shard
#         if self.shuffle:
#             tokens = tokens[torch.randperm(len(tokens))]  # Shuffle the tokens
#         return tokens

#     def next_batch(self):
#         """
#         Fetches the next batch of tokenized data.
#         """
#         # Slice tokens to create input (x) and target (y) sequences
#         buf = self.tokens[self.current_position: self.current_position + self.B * self.T + 1]
#         x = buf[:-1].view(self.B, self.T)  # Inputs (excluding last token)
#         y = buf[1:].view(self.B, self.T)  # Targets (shifted by one)

#         # Advance position for the next batch
#         self.current_position += self.B * self.T * self.num_processes

#         # Check if we need to load the next shard
#         if self.current_position + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
#             self.current_shard = (self.current_shard + 1) % len(self.shards)  # Move to the next shard
#             self.tokens = self._load_and_shuffle_tokens(self.shards[self.current_shard])  # Load and shuffle the new shard
#             self.current_position = self.B * self.T * self.process_rank  # Reset position

#         return x, y  # Return input and target tensors

# Initialize training and validation data loaders
train_loader = DataLoader(B=batch_size_per_device, T=GPTConfig().context_size, 
                          process_rank=local_rank, num_processes=world_size, split="train")
val_loader = DataLoader(B=batch_size_per_device, T=GPTConfig().context_size, 
                        process_rank=local_rank, num_processes=world_size, split="val")

# Set default precision for matrix multiplications
if dtype == 'TF32':
    torch.set_float32_matmul_precision('high')  # Use higher precision for better stability

# Set random seed for reproducibility
if seed is not None:
    torch.manual_seed(seed)  # Seed for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Seed for GPU

enc = tiktoken.get_encoding('gpt2')

# Initialize the model and move it to the appropriate device
model = GPT(GPTConfig()).to(device)  # Create a GPT model using the specified configuration
if compiled:
    model = torch.compile(model)  # Compile model for potential performance improvement (PyTorch 2.0+)

# Configure the optimizer
optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=max_lr, 
                                        betas=(0.9, 0.95), eps=1e-8, device_type=device_type)

if ddp:
    model = DDP(model, device_ids=[local_rank])  # Wrap model in DDP for distributed training
raw_model = model.module if ddp else model  # Access the underlying model when using DDP
model.train()  # Set model to training mode



# Create a log directory and initialize a log file
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:  # Clear the log file
    pass

# Main training loop
for step in range(max_steps):
    last_step = (max_steps - 1 == step)  # Check if this is the last step

    # Evaluation block
    if step % eval_step_interval == 0 or last_step:
        model.eval()  # Set model to evaluation mode
        val_loader.reset()  # Reset the validation loader to the start
        with torch.no_grad():  # Disable gradient computation during evaluation
            val_loss_accum = 0.0  # Accumulate validation loss
            for _ in range(eval_max_steps):
                x, y = val_loader.next_batch()  # Fetch the next batch
                x, y = x.to(device), y.to(device)  # Move data to the appropriate device
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):  # Mixed precision
                    logits, loss = model(x, y)  # Forward pass
                loss = loss / eval_max_steps
                val_loss_accum += loss.detach()  # Accumulate loss

        # Average loss across processes in DDP
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)  # Reduce across processes

        # Log validation loss
        if is_master:
            print(f"Validation Loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

            # Save the model periodically or at the last step
            if step > 0 and (step % save_step == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),  # Save model state
                    'config': raw_model.config,  # Save model configuration
                    'step': step,  # Save current training step
                    'val_loss': val_loss_accum.item()  # Save current validation loss
                }
                torch.save(checkpoint, checkpoint_path)  # Save the checkpoint
    
    # Periodically evaluate the model on the HellaSwag dataset
    if (step % eval_step_interval == 0 or last_step) and (not compiled):
        num_correct_norm = 0  # Counter for correctly classified examples
        num_total = 0  # Counter for total examples processed
        for i, example in enumerate(iterate_examples("val")):  # Iterate over validation examples
            # Ensure only the process with the correct rank handles this example
            if i % world_size != local_rank:
                continue  # Skip this example if it doesn't match the current rank

            # Render the example to extract tokens, mask, and label
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)  # Move tokens to the correct device
            mask = mask.to(device)  # Move mask to the correct device

            # Perform inference to get logits
            with torch.no_grad():  # Disable gradient calculation
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):  # Use mixed precision if applicable
                    logits, loss = model(tokens)  # Forward pass to get logits
                pred_norm = get_most_likely_row(tokens, mask, logits)  # Determine the most likely prediction

            # Update counters
            num_total += 1  # Increment total examples processed
            num_correct_norm += int(pred_norm == label)  # Increment correct predictions if the label matches

        # Reduce and aggregate results across all distributed processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)  # Convert to tensor for reduction
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)  # Sum across processes
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()  # Convert back to integer
            num_correct_norm = num_correct_norm.item()

        # Compute accuracy
        acc_norm = num_correct_norm / num_total  # Calculate normalized accuracy
        if is_master:  # Only the master process logs results
            print(f"HellaSwag Accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:  # Append results to the log file
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # Periodically generate text from the model
    if ((step > 0 and step % eval_step_interval == 0) or last_step) and (not compiled):
        model.eval()  # Set the model to evaluation mode
        num_return_sequences = 4  # Number of sequences to generate
        max_length = 32  # Maximum length of generated sequences
        tokens = enc.encode("Hello, I'm a language model,")  # Encode the input prompt
        tokens = torch.tensor(tokens, dtype=torch.long)  # Convert tokens to a PyTorch tensor
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # Repeat the input for batch processing
        xgen = tokens.to(device)  # Move the input tensor to the correct device

        # Initialize a random number generator for sampling
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + local_rank)  # Seed the generator for reproducibility

        # Generate tokens until the maximum length is reached
        while xgen.size(1) < max_length:
            with torch.no_grad():  # Disable gradient computation
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):  # Use mixed precision if applicable
                    logits, loss = model(xgen)  # Forward pass to get logits
                logits = logits[:, -1, :]  # Extract logits for the last token
                probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities

                # Perform top-k sampling to reduce the sampling space
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # Select the top 50 probabilities
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # Sample a token from the top-k probabilities
                xcol = torch.gather(topk_indices, -1, ix)  # Map sampled indices to token IDs
                xgen = torch.cat((xgen, xcol), dim=1)  # Append the new token to the sequence

        # Decode and print the generated sequences
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()  # Convert generated tokens to a list
            decoded = enc.decode(tokens)  # Decode tokens into text
            print(f"Rank {local_rank} Sample {i}: {decoded}")  # Print the generated text for each sequence

    # Training block
    start_time = time.time()  # Track elapsed time
    optimizer.zero_grad(set_to_none=True)  # Reset gradients
    loss_accum = 0  # Accumulate training loss
    for accum_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()  # Fetch the next training batch
        x, y = x.to(device), y.to(device)  # Move data to the appropriate device

        # Use mixed precision if enabled
        context_manager = torch.autocast(device_type=device, dtype=torch.bfloat16) if fp16 else nullcontext()
        with context_manager:
            logits, loss = model(x, y)  # Forward pass
        loss = loss / grad_accum_steps  # Normalize loss by accumulation steps
        loss_accum += loss.detach()  # Accumulate loss
        if ddp:
            model.require_backward_grad_sync = (accum_step == grad_accum_steps - 1)  # Sync gradients only on last step
        loss.backward()  # Backward pass

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # Average loss across processes

    # Clip gradients to prevent exploding gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update learning rate using the scheduler
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group['lr'] = lr

    # Perform optimizer step
    optimizer.step()

    # Synchronize CUDA operations
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time  # Measure elapsed time for the step
    tokens_per_sec = (train_loader.B * train_loader.T * world_size) / elapsed_time  # Compute tokens/sec

    # Log training progress
    if is_master:
        print(f"Step {step} | Loss = {loss_accum:.4f} | Elapsed Time = {elapsed_time * 1000:.4f} ms | "
              f"Tokens/Sec = {tokens_per_sec:.0f} | Learning Rate = {lr:.5f} | Gradient Norm = {norm:.4f}")

# Destroy the distributed process group when training is complete
if ddp:
    dist.destroy_process_group()