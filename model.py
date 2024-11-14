from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import tiktoken
import time

try:
    print("Path is set to : "  + path)
except NameError:
    path = ""

compiled = True
fp16 = True
dtype = 'TF32'
batch_size_per_device = 16
seed = 1337

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)} (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

@dataclass
class GPTConfig:
    context_size: int = 1024
    vocab_size: int = 50304 # 50257
    num_blocks: int = 12
    num_heads: int = 12
    embed_size: int = 768
    dropout: float = 0.0
    bias: bool = True

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.embed_size, 3 * config.embed_size)
        self.c_proj = nn.Linear(config.embed_size, config.embed_size)
        self.num_heads = config.num_heads
        self.embed_size = config.embed_size
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.context_size, config.context_size)).view(1, 1, config.context_size, config.context_size))
        self.GPT_init_scale = 1

    def forward(self, x):  
        B, T, C = x.size()
        qkv = self.c_attn(x)

        # weighted_dot_product
        q, k, v = qkv.split(self.embed_size, dim=2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1,2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1,2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1,2)

        if self.flash:
          y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
          att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
          att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
          att = F.softmax(att, dim=-1)
          y = att @ v 
      
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embed_size, 4 * config.embed_size, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.embed_size, config.embed_size, bias=config.bias)
        self.GPT_init_scale = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_size, bias=config.bias)
        self.ln_2 = nn.LayerNorm(config.embed_size, bias=config.bias)
        self.attn = CasualSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) 
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_size),
            wpe = nn.Embedding(config.context_size, config.embed_size),
            h = nn.ModuleList([Block(config) for _ in range(config.num_blocks)]),
            ln_f = nn.LayerNorm(config.embed_size, bias=config.bias))
            )

        self.lm_head = nn.Linear(config.embed_size, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self.init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_blocks))
    
    def init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
    
    def forward(self, x, targets=None):
        B, T = x.size()

        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.wpe(pos)
        token_emb = self.transformer.wte(x)
        x = pos_emb + token_emb
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

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

class DataLoader():
    def __init__(self, text_path, B, T):
        self.B = B
        self.T = T
        self.start_pos = 0
        with open(text_path, "r") as file:
            text = file.read()
        tokenizer = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(tokenizer.encode(text))

    def next_batch(self):
        batch = self.tokens[self.start_pos:self.start_pos + self.B * self.T + 1]
        X_batch = batch[:-1].view(self.B, self.T)
        Y_batch = batch[1:].view(self.B, self.T)

        self.start_pos += self.B * self.T

        if self.start_pos + self.B * self.T + 1 > len(self.tokens):
            self.start_pos = 0
        
        return X_batch, Y_batch


model = GPT.from_pretrained('gpt2')
model.eval()
model.to(device)

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(5, 1).to(device)

max_length=30

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

dataloader = DataLoader(path+"input.txt", B=batch_size_per_device, T=GPTConfig().context_size)

max_train_steps = 50
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

for step in range(max_train_steps):
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    x, y = dataloader.next_batch()
    x, y = x.to(device), y.to(device)
    context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if fp16 else nullcontext()
    with context_manager:
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    token_per_sec = (dataloader.B * dataloader.T) / elapsed_time 
    print(f"Step {step} | Loss = {loss:.4f} | Elapsed Time = {elapsed_time * 1000:.4f} ms | Token/Sec = {token_per_sec:.0f} tokens | Learning Rate = {lr}, Gradient Normal = {norm:.4f}")

'''add scheduler and set up new optimizer set up
grad accum and fix scaling
ddp and modify dataloader'''
