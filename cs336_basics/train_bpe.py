"""BPE Training implementation."""

import os
import regex
from collections import defaultdict


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer from a corpus.

    Args:
        input_path: Path to the training corpus file.
        vocab_size: Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: A list of string special tokens to be added to the tokenizer vocabulary.

    Returns:
        tuple containing:
            vocab: The trained tokenizer vocabulary, a mapping from int (token ID) to bytes.
            merges: BPE merges, ordered by order of creation.
    """
    # Read the corpus
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Initialize vocabulary with special tokens first
    vocab: dict[int, bytes] = {}
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode("utf-8")
    
    # Add all 256 single bytes
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    
    # Build special token pattern to remove them from text before training
    if special_tokens:
        # Sort by length (descending) to match longer special tokens first
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped = [regex.escape(token) for token in sorted_special_tokens]
        special_pattern = regex.compile("(" + "|".join(escaped) + ")")
        # Remove special tokens from training text
        text = special_pattern.sub("", text)
    
    # GPT-2 pre-tokenization pattern
    pat = regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    
    # Pre-tokenize and count word frequencies
    pre_tokens = pat.findall(text)
    
    # Count each pre-token's frequency
    word_counts: dict[bytes, int] = defaultdict(int)
    for pre_token in pre_tokens:
        word_bytes = pre_token.encode("utf-8")
        word_counts[word_bytes] += 1
    
    # Convert each word to a list of single-byte tokens
    # word_splits maps word_bytes -> list of tokens (each token is bytes)
    word_splits: dict[bytes, list[bytes]] = {}
    for word in word_counts:
        word_splits[word] = [bytes([b]) for b in word]
    
    merges: list[tuple[bytes, bytes]] = []
    
    # Keep merging until we reach the target vocab size
    while len(vocab) < vocab_size:
        # Count all adjacent pairs across all words, weighted by word frequency
        pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        
        for word, tokens in word_splits.items():
            if len(tokens) < 2:
                continue
            count = word_counts[word]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += count
        
        if not pair_counts:
            break
        
        # Find the most frequent pair
        # When there's a tie, choose the pair with highest lexicographic order
        max_count = max(pair_counts.values())
        best_pairs = [p for p, c in pair_counts.items() if c == max_count]
        best_pair = max(best_pairs)  # lexicographic order
        
        # Add the merge
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        
        # Update all word splits by merging this pair
        for word in word_splits:
            tokens = word_splits[word]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            word_splits[word] = new_tokens
    
    return vocab, merges
