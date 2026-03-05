"""BPE Tokenizer implementation."""

import regex
from typing import Iterable


class Tokenizer:
    """BPE Tokenizer implementation compatible with GPT-2 style tokenization."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initialize the BPE tokenizer.

        Args:
            vocab: A mapping from token ID (int) to token bytes.
            merges: A list of BPE merge tuples (token1_bytes, token2_bytes), ordered by priority.
            special_tokens: A list of special token strings that should never be split.
        """
        self.vocab = vocab
        self.merges = merges
        
        # Build reverse vocab mapping: bytes -> token ID
        self.vocab_reverse: dict[bytes, int] = {v: k for k, v in vocab.items()}
        
        # Build merge ranking for efficient lookup
        # Lower rank = higher priority (should be merged first)
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            merge: i for i, merge in enumerate(merges)
        }
        
        # Handle special tokens
        self.special_tokens = special_tokens if special_tokens else []
        
        # Build special token pattern for splitting
        if self.special_tokens:
            # Sort by length (descending) to match longer special tokens first
            # This ensures that overlapping tokens like "<|endoftext|>" and "<|endoftext|><|endoftext|>"
            # will match the longer one first
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # Escape special regex characters and join with |
            escaped = [regex.escape(token) for token in sorted_special_tokens]
            self.special_pattern = regex.compile("(" + "|".join(escaped) + ")")
        else:
            self.special_pattern = None
        
        # GPT-2 pre-tokenization pattern
        # This pattern matches:
        # - 's, 't, 're, 've, 'm, 'll, 'd (contractions)
        # - \s+[^\s]? (whitespace sequences, with optional non-whitespace)
        # - \p{L}+ (Unicode letters)
        # - \p{N}+ (Unicode numbers)
        # - [^\s\p{L}\p{N}]+ (other non-whitespace, non-letter, non-number)
        self.pat = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
    
    def _tokenize_chunk(self, chunk: str) -> list[int]:
        """
        Tokenize a chunk of text (without special tokens) using BPE.
        
        Args:
            chunk: A string chunk to tokenize.
            
        Returns:
            A list of token IDs.
        """
        if not chunk:
            return []
        
        # Pre-tokenize using GPT-2 pattern
        pre_tokens = self.pat.findall(chunk)
        
        result_ids = []
        
        for pre_token in pre_tokens:
            # Convert to bytes
            pre_token_bytes = pre_token.encode("utf-8")
            
            # Convert to list of single-byte tokens
            # Each element is a bytes object representing a token
            tokens = [bytes([b]) for b in pre_token_bytes]
            
            # Apply BPE merges iteratively
            while len(tokens) > 1:
                # Find the best merge (lowest rank)
                best_merge = None
                best_rank = float("inf")
                best_idx = -1
                
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merge_ranks:
                        rank = self.merge_ranks[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_merge = pair
                            best_idx = i
                
                # No more merges possible
                if best_merge is None:
                    break
                
                # Apply the merge
                new_token = best_merge[0] + best_merge[1]
                tokens = tokens[:best_idx] + [new_token] + tokens[best_idx + 2:]
            
            # Convert tokens to IDs
            for token in tokens:
                if token in self.vocab_reverse:
                    result_ids.append(self.vocab_reverse[token])
                else:
                    # This shouldn't happen with a complete vocab, but handle gracefully
                    # by encoding each byte individually
                    for b in token:
                        single_byte = bytes([b])
                        if single_byte in self.vocab_reverse:
                            result_ids.append(self.vocab_reverse[single_byte])
        
        return result_ids
    
    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs.
        
        Args:
            text: The input string to encode.
            
        Returns:
            A list of token IDs.
        """
        if not text:
            return []
        
        result_ids = []
        
        if self.special_pattern:
            # Split by special tokens, keeping them in the result
            parts = self.special_pattern.split(text)
            
            for part in parts:
                if not part:
                    continue
                
                if part in self.special_tokens:
                    # This is a special token - encode directly
                    special_bytes = part.encode("utf-8")
                    if special_bytes in self.vocab_reverse:
                        result_ids.append(self.vocab_reverse[special_bytes])
                else:
                    # Regular text - tokenize with BPE
                    result_ids.extend(self._tokenize_chunk(part))
        else:
            # No special tokens, just tokenize the whole text
            result_ids.extend(self._tokenize_chunk(text))
        
        return result_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        Encode an iterable of strings into token IDs, yielding them one at a time.
        
        This method is memory-efficient for large inputs as it doesn't load
        the entire input into memory at once.
        
        Args:
            iterable: An iterable of strings (e.g., a file handle).
            
        Yields:
            Token IDs one at a time.
        """
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.
        
        Args:
            ids: A list of token IDs.
            
        Returns:
            The decoded string.
        """
        # Concatenate all token bytes
        token_bytes = b""
        for token_id in ids:
            if token_id in self.vocab:
                token_bytes += self.vocab[token_id]
        
        # Decode to string, replacing errors
        return token_bytes.decode("utf-8", errors="replace")
