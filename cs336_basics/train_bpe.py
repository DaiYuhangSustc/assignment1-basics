"""BPE Training implementation with aggressive memory optimization using pickle."""

import os
import pickle
import tempfile
import regex
import gc
from collections import defaultdict


def _save_to_pickle(data: object, file_path: str) -> None:
    """Save data to a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_from_pickle(file_path: str) -> object:
    """Load data from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _count_words_streaming(
    input_path: str | os.PathLike,
    pat: regex.Pattern,
    special_pattern: regex.Pattern | None = None,
    chunk_size: int = 10 * 1024 * 1024,  # 10MB chunks
) -> dict[bytes, int]:
    """
    Count word frequencies using streaming/chunked reading.
    
    This processes the file in chunks to avoid loading everything into memory.
    Uses a buffer approach to handle pre-tokens that might span chunk boundaries.
    """
    word_counts: dict[bytes, int] = defaultdict(int)
    
    with open(input_path, "r", encoding="utf-8") as f:
        # Use a buffer to handle pre-tokens that might span chunks
        buffer = ""
        overlap_size = 1000  # Keep last 1000 chars to handle split tokens
        
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            buffer += chunk
            
            # Remove special tokens from buffer if needed
            if special_pattern:
                buffer = special_pattern.sub("", buffer)
            
            # Find all pre-tokens
            pre_tokens = pat.findall(buffer)
            
            # Count pre-tokens (except possibly the last one which might be incomplete)
            # We keep the last incomplete token in the buffer
            for pre_token in pre_tokens[:-1]:  # Exclude last token
                word_bytes = pre_token.encode("utf-8")
                word_counts[word_bytes] += 1
            
            # Keep the last token and some overlap for next iteration
            if pre_tokens:
                buffer = pre_tokens[-1]  # Keep last token (might be incomplete)
            else:
                buffer = ""
            
            # Also keep some raw text as overlap
            if len(buffer) < overlap_size:
                # Get the last part of the original buffer
                original_end = buffer[-overlap_size:] if len(buffer) > overlap_size else buffer
                buffer = original_end
        
        # Process remaining buffer
        if buffer:
            if special_pattern:
                buffer = special_pattern.sub("", buffer)
            pre_tokens = pat.findall(buffer)
            for pre_token in pre_tokens:
                word_bytes = pre_token.encode("utf-8")
                word_counts[word_bytes] += 1
    
    return dict(word_counts)


def _get_pair_counts(
    word_splits: dict[bytes, list[bytes]],
    word_counts: dict[bytes, int],
) -> dict[tuple[bytes, bytes], int]:
    """Calculate pair counts from word splits."""
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    
    for word, tokens in word_splits.items():
        if len(tokens) < 2:
            continue
        count = word_counts[word]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += count
    
    return dict(pair_counts)


def _merge_pair_in_word_splits(
    word_splits: dict[bytes, list[bytes]],
    best_pair: tuple[bytes, bytes],
    new_token: bytes,
) -> dict[bytes, list[bytes]]:
    """Merge the best pair in all word splits."""
    new_word_splits = {}
    
    for word, tokens in word_splits.items():
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        new_word_splits[word] = new_tokens
    
    return new_word_splits


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    use_pickle_cache: bool = True,
    pickle_cache_dir: str | None = None,
    clear_cache_on_exit: bool = True,
    cache_threshold: int = 10000,
    streaming_read: bool = True,
    chunk_size: int = 10 * 1024 * 1024,  # 10MB
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer from a corpus with memory optimization.

    Args:
        input_path: Path to the training corpus file.
        vocab_size: Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: A list of string special tokens to be added to the tokenizer vocabulary.
        use_pickle_cache: Whether to use pickle files to cache word_splits during merge iterations (default: True).
        pickle_cache_dir: Directory to store pickle cache files (default: system temp dir).
        clear_cache_on_exit: Whether to delete pickle cache files when done (default: True).
        cache_threshold: Minimum number of unique words before using pickle cache (default: 10000).
        streaming_read: Whether to use streaming file reading (default: True, recommended for large files).
        chunk_size: Size of chunks when streaming (default: 10MB).

    Returns:
        tuple containing:
            vocab: The trained tokenizer vocabulary, a mapping from int (token ID) to bytes.
            merges: BPE merges, ordered by order of creation.
    """
    word_splits_cache_path = None
    word_counts_cache_path = None
    
    if pickle_cache_dir is None:
        pickle_cache_dir = tempfile.gettempdir()
    
    # Create unique cache file names
    pid = os.getpid()
    word_splits_cache_path = os.path.join(pickle_cache_dir, f"bpe_word_splits_{pid}.pkl")
    word_counts_cache_path = os.path.join(pickle_cache_dir, f"bpe_word_counts_{pid}.pkl")
    
    try:
        # Initialize vocabulary with special tokens first
        vocab: dict[int, bytes] = {}
        for i, token in enumerate(special_tokens):
            vocab[i] = token.encode("utf-8")
        
        # Add all 256 single bytes
        for i in range(256):
            vocab[len(vocab)] = bytes([i])
        
        # Build special token pattern to remove them from text before training
        special_pattern = None
        if special_tokens:
            # Sort by length (descending) to match longer special tokens first
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            escaped = [regex.escape(token) for token in sorted_special_tokens]
            special_pattern = regex.compile("(" + "|".join(escaped) + ")")
        
        # GPT-2 pre-tokenization pattern
        pat = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        
        # Count word frequencies - use streaming or full read based on parameter
        if streaming_read:
            print("Counting word frequencies (streaming mode)...")
            word_counts = _count_words_streaming(input_path, pat, special_pattern, chunk_size)
        else:
            print("Counting word frequencies (full read mode)...")
            # Read the entire corpus
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Remove special tokens from training text
            if special_pattern:
                text = special_pattern.sub("", text)
            
            # Pre-tokenize and count word frequencies
            pre_tokens = pat.findall(text)
            
            # Count each pre-token's frequency
            word_counts: dict[bytes, int] = defaultdict(int)
            for pre_token in pre_tokens:
                word_bytes = pre_token.encode("utf-8")
                word_counts[word_bytes] += 1
            word_counts = dict(word_counts)
            
            # Free memory
            del text
            del pre_tokens
            gc.collect()
        
        print(f"Found {len(word_counts)} unique words")
        
        # Convert each word to a list of single-byte tokens
        print("Building initial word splits...")
        word_splits: dict[bytes, list[bytes]] = {}
        for word in word_counts:
            word_splits[word] = [bytes([b]) for b in word]
        
        # Determine if we should use pickle cache based on data size
        use_cache = use_pickle_cache and len(word_splits) >= cache_threshold
        
        if use_cache:
            print(f"Using pickle cache (threshold: {cache_threshold})")
            # Cache word_counts and word_splits to pickle files to reduce memory pressure
            _save_to_pickle(word_counts, word_counts_cache_path)
            _save_to_pickle(word_splits, word_splits_cache_path)
            print(f"Cached word_counts to: {word_counts_cache_path}")
            print(f"Cached word_splits to: {word_splits_cache_path}")
            # Clear from memory
            del word_counts
            del word_splits
            word_counts = None
            word_splits = None
            gc.collect()
            print("Cleared data from memory, will load from cache during iterations")
        
        merges: list[tuple[bytes, bytes]] = []
        
        # Keep merging until we reach the target vocab size
        iteration = 0
        while len(vocab) < vocab_size:
            iteration += 1
            
            # Load data from cache if using pickle cache
            if use_cache:
                word_counts = _load_from_pickle(word_counts_cache_path)
                word_splits = _load_from_pickle(word_splits_cache_path)
            
            # Count all adjacent pairs across all words, weighted by word frequency
            pair_counts = _get_pair_counts(word_splits, word_counts)
            
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
            word_splits = _merge_pair_in_word_splits(word_splits, best_pair, new_token)
            
            # Save updated word_splits back to pickle cache and clear from memory
            if use_cache:
                _save_to_pickle(word_splits, word_splits_cache_path)
                _save_to_pickle(word_counts, word_counts_cache_path)
                # Clear from memory
                del word_counts
                del word_splits
                word_counts = None
                word_splits = None
                gc.collect()
            
            # Progress logging
            if iteration % 100 == 0:
                print(f"Merge iteration {iteration}, vocab size: {len(vocab)}")
        
        # Final cleanup
        if use_cache:
            # Load one final time to return (but we don't need word_counts anymore)
            pass
        
        return vocab, merges
    
    finally:
        # Clean up pickle cache files
        if clear_cache_on_exit:
            if word_splits_cache_path and os.path.exists(word_splits_cache_path):
                try:
                    os.remove(word_splits_cache_path)
                    print(f"Cleaned up cache: {word_splits_cache_path}")
                except OSError as e:
                    print(f"Warning: Could not remove cache file: {e}")
            if word_counts_cache_path and os.path.exists(word_counts_cache_path):
                try:
                    os.remove(word_counts_cache_path)
                    print(f"Cleaned up cache: {word_counts_cache_path}")
                except OSError as e:
                    print(f"Warning: Could not remove cache file: {e}")
