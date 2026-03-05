"""
Training script for a minimal language model.

This script combines:
- Tokenizer: for text <-> token conversion
- Model: TransformerLM for language modeling
- Optimizer: AdamW for parameter updates
- Data: batch loading utilities
"""

import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe
from cs336_basics.serialization import save_checkpoint, load_checkpoint


def train_minimal_lm(
    # Model hyperparameters
    vocab_size: int = 512,
    context_length: int = 128,
    d_model: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    d_ff: int = 512,
    rope_theta: float = 10000.0,
    # Training hyperparameters
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    num_epochs: int = 3,
    grad_clip_norm: float = 1.0,
    # Data
    data_path: str = "data/TinyStoriesV2-GPT4-train.txt",
    # Checkpointing
    checkpoint_dir: str = "checkpoints",
    save_every: int = 100,
):
    """
    Train a minimal language model.
    
    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        rope_theta: RoPE theta parameter
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        grad_clip_norm: Gradient clipping norm
        data_path: Path to training data
        checkpoint_dir: Directory to save checkpoints
        save_every: Save checkpoint every N steps
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # =========================================================================
    # 1. Load and tokenize data
    # =========================================================================
    print("Loading data...")
    
    # Special tokens for the tokenizer
    special_tokens = ["<|endoftext|>"]
    
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found. Using synthetic data for demonstration.")
        # Create synthetic text data for demonstration
        text = "Once upon a time there was a little girl who loved to play in the garden. "
        text = text * 1000  # Repeat to get enough data
        
        # Train a simple BPE tokenizer on this data
        # First, save to a temp file
        temp_path = "/tmp/synthetic_text.txt"
        with open(temp_path, "w") as f:
            f.write(text)
        
        # Train tokenizer
        vocab, merges = train_bpe(temp_path, vocab_size=256, special_tokens=special_tokens)
        tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)
        
        # Tokenize
        token_ids = tokenizer.encode(text)
    else:
        # Train tokenizer on the data
        print("Training BPE tokenizer...")
        vocab, merges = train_bpe(data_path, vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)
        
        # Read and tokenize data
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Use only first portion of data to speed up
        max_train_tokens = 100000
        token_ids = tokenizer.encode(text[:50000])  # Limit text length for speed
    
    dataset = np.array(token_ids, dtype=np.int64)
    print(f"Dataset size: {len(dataset)} tokens")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    # =========================================================================
    # 2. Initialize model
    # =========================================================================
    print("Initializing model...")
    
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=torch.float32,
    )
    
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # =========================================================================
    # 3. Initialize optimizer
    # =========================================================================
    print("Initializing optimizer...")
    
    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    # =========================================================================
    # 4. Training loop
    # =========================================================================
    print("Starting training...")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    def format_time(seconds: float) -> str:
        """Format seconds into a readable time string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins, secs = divmod(int(seconds), 60)
            return f"{mins}m {secs}s"
        else:
            hours, remainder = divmod(int(seconds), 3600)
            mins, secs = divmod(remainder, 60)
            return f"{hours}h {mins}m {secs}s"
    
    def print_progress_bar(
        iteration: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
        length: int = 30,
        fill: str = "█",
    ):
        """Print a progress bar that refreshes in place."""
        percent = 100 * (iteration / float(total)) if total > 0 else 100
        filled_length = int(length * iteration // total) if total > 0 else length
        bar = fill * filled_length + "-" * (length - filled_length)
        sys.stdout.write(f"\r{prefix} |{bar}| {iteration}/{total} ({percent:.1f}%) {suffix}")
        sys.stdout.flush()
    
    iteration = 0
    # Calculate total iterations for progress tracking
    num_samples = len(dataset) - context_length - 1
    batches_per_epoch = (num_samples + batch_size - 1) // batch_size if num_samples > 0 else 1
    total_iterations = num_epochs * batches_per_epoch
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Shuffle dataset each epoch
        indices = np.random.permutation(num_samples)
        
        epoch_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        for batch_idx in range(0, len(indices), batch_size):
            # Get batch
            batch_start_indices = indices[batch_idx:batch_idx + batch_size]
            
            # Build batch manually (simple version)
            max_start_idx = len(dataset) - context_length - 1
            starting_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
            
            x = np.array([dataset[i:i + context_length] for i in starting_indices])
            y = np.array([dataset[i + 1:i + context_length + 1] for i in starting_indices])
            
            x = torch.tensor(x, dtype=torch.long, device=device)
            y = torch.tensor(y, dtype=torch.long, device=device)
            
            # =========================================================================
            # Forward pass
            # =========================================================================
            optimizer.zero_grad()
            
            # Get logits: (batch_size, seq_len, vocab_size)
            logits = model(x)
            
            # Compute loss (shifted by 1 for next-token prediction)
            # Flatten for cross entropy: (batch_size * seq_len, vocab_size)
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = y.view(-1)
            
            loss = cross_entropy(logits_flat, targets_flat)
            
            # =========================================================================
            # Backward pass
            # =========================================================================
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            # Update parameters
            optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            iteration += 1
            
            # Calculate elapsed time and ETA
            elapsed = time.time() - start_time
            avg_time_per_iter = elapsed / iteration if iteration > 0 else 0
            remaining_iters = total_iterations - iteration
            eta = avg_time_per_iter * remaining_iters
            
            # Update progress bar
            current_batch = batch_idx // batch_size + 1
            avg_loss = epoch_loss / num_batches
            print_progress_bar(
                iteration=iteration,
                total=total_iterations,
                prefix=f"Epoch {epoch + 1}/{num_epochs}",
                suffix=f"Loss: {avg_loss:.4f} | Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}",
            )
            
            # Save checkpoint
            if iteration % save_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.pt")
                save_checkpoint(model, optimizer, iteration, checkpoint_path)
                # Print checkpoint message on a new line, then continue progress bar
                print(f"\nCheckpoint saved: {checkpoint_path}")
        
        # Epoch summary - print on new line
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed | Average Loss: {avg_epoch_loss:.4f} | Time: {format_time(epoch_time)}")
    
    # =========================================================================
    # 5. Save final model
    # =========================================================================
    final_checkpoint = os.path.join(checkpoint_dir, "final_model.pt")
    save_checkpoint(model, optimizer, iteration, final_checkpoint)
    print(f"Training complete! Final model saved to: {final_checkpoint}")
    
    return model, optimizer, tokenizer


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    device: str = "cpu",
) -> str:
    """
    Generate text using the trained model.
    
    Args:
        model: Trained TransformerLM
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, sample only from top k tokens
        device: Device to run generation on
    
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    # Get context_length from model's RoPE max_seq_len
    # We need to truncate to avoid index out of bounds in RoPE
    context_length = model.context_length
    
    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate input to context_length if needed
            if input_ids.shape[1] > context_length:
                input_ids = input_ids[:, -context_length:]
            
            # Forward pass
            logits = model(input_ids)
            
            # Get logits for last position
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits_filtered = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits_filtered[top_k_indices] = top_k_values
                next_token_logits = next_token_logits_filtered
            
            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we hit EOS (if tokenizer has EOS)
            # For simplicity, we just continue
    
    # Decode
    generated_ids = input_ids[0].cpu().numpy()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


if __name__ == "__main__":
    # Train a minimal language model
    model, optimizer, tokenizer = train_minimal_lm(
        # Small model for quick training
        vocab_size=256,      # Byte-level vocabulary
        context_length=64,   # Short context
        d_model=128,         # Small model dimension
        num_layers=2,        # Few layers
        num_heads=2,        # Few heads
        d_ff=256,           # Small FFN
        rope_theta=10000.0,
        # Training settings
        batch_size=16,
        learning_rate=1e-3,
        num_epochs=1,
        grad_clip_norm=1.0,
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    
    # Demo generation
    try:
        prompt = "Once upon a time"
        print(f"\nGenerating from prompt: '{prompt}'")
        
        # Note: Generation requires the model to be on the same device
        device = next(model.parameters()).device
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=50,
            temperature=0.8,
            device=str(device),
        )
        print(f"Generated: {generated}")
    except Exception as e:
        print(f"Generation skipped: {e}")
