from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import tiktoken
import time
import inspect
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Path for input data and saving the model
path = "/content/drive/MyDrive/GPT2/"

# Configuration for the GPT model
@dataclass
class GPTConfig:
    context_size: int = 1024  # Sequence length
    vocab_size: int = 50304  # Vocabulary size (e.g., GPT-2 uses 50257)
    num_blocks: int = 12  # Number of transformer blocks
    num_heads: int = 12  # Number of attention heads
    embed_size: int = 768  # Embedding size
    dropout: float = 0.0  # Dropout rate
    bias: bool = True  # Whether to use biases in linear layers

# Check if Distributed Data Parallel (DDP) is enabled
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    # Initialize DDP settings
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    is_master = (local_rank == 0)  # Master rank for logging
    if is_master:
        print("Distributed learning ON")
        print(f"Global Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
else:
    # Fallback for single-device training
    rank, local_rank, world_size = 0, 0, 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_master = True

    # Log the device being used
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)} (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS")
    else:
        print("Using device: CPU")

# Global training configurations
compiled = True  # Use `torch.compile` if available
fp16 = True  # Mixed precision training flag
dtype = 'TF32'  # Default matmul precision
total_batch_size = 524288  # Total batch size
batch_size_per_device = 16  # Batch size for each GPU
seed = 1337  # Random seed for reproducibility

# Calculate gradient accumulation steps
grad_accum_steps = total_batch_size // (batch_size_per_device * GPTConfig().context_size * world_size)

# Define the GPT model components
class CasualSelfAttention(nn.Module):
    """
    Implements scaled dot-product attention with optional dropout and causal masking.
    """
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.embed_size, 3 * config.embed_size)  # Query, Key, Value projections
        self.c_proj = nn.Linear(config.embed_size, config.embed_size)  # Output projection
        self.num_heads = config.num_heads
        self.embed_size = config.embed_size
        self.dropout = config.dropout
        self.att_dropout = nn.Dropout(config.dropout)  # Attention dropout
        self.residual_dropout = nn.Dropout(config.dropout)  # Residual dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask for slower attention implementation
            self.register_buffer("bias", torch.tril(torch.ones(config.context_size, config.context_size))
                                  .view(1, 1, config.context_size, config.context_size))

    def forward(self, x):
        B, T, C = x.size()  # Batch, sequence length, embedding size
        qkv = self.c_attn(x)  # Linear projection
        q, k, v = qkv.split(self.embed_size, dim=2)  # Split into query, key, and value

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        if self.flash:  # Fast attention path
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0)
        else:  # Slow attention path
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            if self.training:
                att = self.att_dropout(att)
            y = att @ v

        # Combine heads and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        if self.training:
            y = self.residual_dropout(y)
        return y

class MLP(nn.Module):
    """
    Multi-layer perceptron used in transformer blocks.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embed_size, 4 * config.embed_size, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')  # GeLU activation
        self.c_proj = nn.Linear(4 * config.embed_size, config.embed_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        if self.training:
            x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_size, bias=config.bias)  # Layer normalization
        self.ln_2 = nn.LayerNorm(config.embed_size, bias=config.bias)
        self.attn = CasualSelfAttention(config)  # Attention layer
        self.mlp = MLP(config)  # Feed-forward layer

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # Apply attention with residual connection
        x = x + self.mlp(self.ln_2(x))  # Apply MLP with residual connection
        return x

class GPT(nn.Module):
    """
    GPT-like model implementation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Transformer components
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_size),  # Token embeddings
            wpe = nn.Embedding(config.context_size, config.embed_size),  # Positional embeddings
            dropout = nn.Dropout(config.dropout),  # Dropout layer
            h = nn.ModuleList([Block(config) for _ in range(config.num_blocks)]),  # Transformer blocks
            ln_f = nn.LayerNorm(config.embed_size, bias=config.bias))  # Final normalization
        )

        self.lm_head = nn.Linear(config.embed_size, config.vocab_size, bias=False)  # Output layer

        # Tie input embedding weights with the output layer
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self.init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_blocks))

    def init_weights(self, module):
        """
        Initialize weights for the model layers.
        """
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, x, targets=None):
        """
        Forward pass for training and inference.
        """
        B, T = x.size()  # Batch size and sequence length

        # Combine token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.wpe(pos)
        token_emb = self.transformer.wte(x)
        x = pos_emb + token_emb
        if self.training:
            x = self.transformer.dropout(x)

        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)  # Final layer normalization

        # Compute loss for training or logits for inference
        loss = None
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])  # Predict next token logits

        return logits, loss

    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens given input tokens.
        """
        with torch.no_grad():
            for i in range(max_new_tokens):
                tokens_cropped = tokens if tokens.size(1) <= self.config.context_size else tokens[:, -self.config.context_size:]
                logits, _ = self(tokens_cropped)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:  # Apply top-k sampling
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
                new_token = torch.multinomial(probs, num_samples=1)  # Sample next token
                tokens = torch.cat((tokens, new_token), dim=1)

        return tokens

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(num_blocks=12, num_heads=12, embed_size=768),  # 124M params
            'gpt2-medium':  dict(num_blocks=24, num_heads=16, embed_size=1024), # 350M params
            'gpt2-large':   dict(num_blocks=36, num_heads=20, embed_size=1280), # 774M params
            'gpt2-xl':      dict(num_blocks=48, num_heads=25, embed_size=1600), # 1558M params
        }[model_type]
        print("Forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['context_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

max_steps = 500
warmup_steps = 100

max_lr = 6e-4
min_lr = max_lr * 0.1

def get_lr(step):
    if step < warmup_steps:
        return ((step+1) / warmup_steps) * max_lr

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)

class DataLoader():
    def __init__(self, text_path, B, T, rank, world_size):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size

        self.start_pos = self.B * self.T * self.rank

        with open(text_path, "r") as file:
            text = file.read()
        tokenizer = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(tokenizer.encode(text))

    def next_batch(self):
        batch = self.tokens[self.start_pos:self.start_pos + self.B * self.T + 1]
        X_batch = batch[:-1].view(self.B, self.T)
        Y_batch = batch[1:].view(self.B, self.T)

        self.start_pos += self.B * self.T * self.world_size

        if self.start_pos + (self.B * self.T * self.world_size + 1) > len(self.tokens):
            self.start_pos = self.B * self.T * self.rank

        return X_batch, Y_batch


model = GPT.from_pretrained('gpt2')
model.eval()
model.to(device)

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(5, 1).to(device)

max_length=30

if seed is not None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, 50, dim=-1)
        next_token = torch.multinomial(top_k_probs, 1)
        next_token_indices = torch.gather(top_k_indices, -1, next_token)
        x = torch.cat((x, next_token_indices), dim=1)

outputs = [enc.decode(list(output)) for output in x]

for output in outputs:
    print(output)

if dtype == 'TF32':
    torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device)

if compiled:
    model = torch.compile(model)

dataloader = DataLoader(path+"input.txt", B=batch_size_per_device, T=GPTConfig().context_size, rank=local_rank, world_size=world_size)

model.train()

max_train_steps = 50
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.01, learning_rate=max_lr, betas=(0.9, 0.95), eps=1e-8, device_type=device)
if ddp:
    model = DDP(model, device_ids=[local_rank])

for step in range(max_train_steps):
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0
    for accum_step in range(grad_accum_steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if fp16 else nullcontext()
        with context_manager:
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (accum_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    token_per_sec = (dataloader.B * dataloader.T * world_size) / elapsed_time
    print(f"Step {step} | Loss = {loss_accum:.4f} | Elapsed Time = {elapsed_time * 1000:.4f} ms | Token/Sec = {token_per_sec:.0f} tokens | Learning Rate = {lr:.5f}, Gradient Normal = {norm:.4f}")

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(5, 1).to(device)

model.eval()
outputs = model.generate(x, max_new_tokens=30)
outputs = [enc.decode(list(output)) for output in outputs]

for text in outputs:
    print(text)
    
if ddp:
    dist.destroy_process_group()

'''start dataset running'''

