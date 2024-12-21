# Required Libraries
from contextlib import nullcontext  # For conditional context management
import torch  # PyTorch library for building and training models
import torch.nn as nn  # Neural network module in PyTorch
import torch.nn.functional as F  # Functional interface for neural network operations
from dataclasses import dataclass  # To create simple data structures
import math  # For mathematical computations
import tiktoken  # For tokenizing input text
import time  # To measure execution time
import os  # For file and environment variable operations
import torch.distributed as dist  # For distributed training
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP module for multi-GPU training
from hellaswag import render_example, iterate_examples  # Specific dataset functions
from model import GPT, GPTConfig, DataLoader, get_most_likely_row
import argparse
from contextlib import nullcontext, contextmanager

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument("--path", type=str, default="", help="Path for input data and saving the model")
    parser.add_argument("--max_steps", type=int, default=19073, help="Total number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=715, help="Warmup steps for learning rate scheduler")
    parser.add_argument("--max_lr", type=float, default=6e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate")
    parser.add_argument("--batch_size_per_device", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--total_batch_size", type=int, default=524288, help="Total batch size (tokens)")
    parser.add_argument("--eval_step_interval", type=int, default=250, help="Steps between evaluations")
    parser.add_argument("--eval_max_steps", type=int, default=20, help="Max steps for evaluation")
    parser.add_argument("--save_step", type=int, default=5000, help="Save model every this many steps")
    parser.add_argument("--seed", type=int, default=1337, help="Seed for reproducibility")
    parser.add_argument("--compiled", action="store_true", help="Use PyTorch compiled mode")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed-precision training")
    return parser.parse_args()

# Learning rate scheduler using a cosine decay with warmup
def get_lr(step, warmup_steps, max_lr, min_lr, max_steps):
    if step < warmup_steps:
        return ((step + 1) / warmup_steps) * max_lr  # Linear warmup
    if step > max_steps:
        return min_lr  # Return minimum LR after decay

    # Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine function
    return min_lr + coeff * (max_lr - min_lr)  # Adjust learning rate based on cosine decay

def main():
    args = parse_arguments()

    path = args.path
    warmup_steps = args.warmup_steps
    max_steps = args.max_steps
    max_lr = args.max_lr
    min_lr = args.min_lr
    batch_size_per_device = args.batch_size_per_device
    total_batch_size = args.total_batch_size
    eval_step_interval = args.eval_step_interval
    eval_max_steps = args.eval_max_steps
    save_step = args.save_step
    seed = args.seed
    compiled = args.compiled
    fp16 = args.fp16

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

    # Calculate gradient accumulation steps
    grad_accum_steps = total_batch_size // (batch_size_per_device * GPTConfig().context_size * world_size)
    if is_master:
        print("The desired amount of tokens in a single batch is:", total_batch_size)
        print("Gradient Accumulation Steps:", grad_accum_steps)

    # Initialize training and validation data loaders
    train_loader = DataLoader(B=batch_size_per_device, T=GPTConfig().context_size, 
                            process_rank=local_rank, num_processes=world_size, split="train", is_master=is_master, path=path)
    val_loader = DataLoader(B=batch_size_per_device, T=GPTConfig().context_size, 
                            process_rank=local_rank, num_processes=world_size, split="val", is_master=is_master, path=path)

    torch.set_float32_matmul_precision('high')  # Use higher precision for better stability

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)  # Seed for CPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # Seed for GPU

    enc = tiktoken.get_encoding('gpt2')

    # Initialize the model and move it to the appropriate device
    model = GPT(GPTConfig(), is_master=is_master).to(device)  # Create a GPT model using the specified configuration
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
    log_dir = path+"log"
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
                    context_manager = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if fp16 else nullcontext()
                    with context_manager:  # Mixed precision
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
                    context_manager = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if fp16 else nullcontext()
                    with context_manager:  # Use mixed precision if applicable
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
                    context_manager = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if fp16 else nullcontext()
                    with context_manager:  # Use mixed precision if applicable
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
        lr = get_lr(step, warmup_steps, max_lr, min_lr, max_steps)
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

if __name__ == "__main__":
    main()