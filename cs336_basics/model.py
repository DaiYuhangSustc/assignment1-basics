"""
Transformer Language Model Architecture
Implementation from scratch following the assignment requirements.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(nn.Module):
    """
    Linear transformation layer without bias.
    y = W @ x where W is of shape (d_out, d_in)
    """
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module.
        
        Args:
            in_features: final dimension of the input
            out_features: final dimension of the output
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with truncated normal
        # std = sqrt(2 / (d_in + d_out))
        std = math.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x: Input tensor of shape (..., in_features)
        
        Returns:
            Output tensor of shape (..., out_features)
        """
        # x: (..., in_features)
        # weight: (out_features, in_features)
        # output: (..., out_features)
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """
    Embedding layer that maps integer token IDs to dense vectors.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Construct an embedding module.
        
        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of the embedding vectors (d_model)
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize with truncated normal, std=1
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Args:
            token_ids: Tensor of shape (...) containing integer token IDs
        
        Returns:
            Tensor of shape (..., embedding_dim) with embedding vectors
        """
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    RMSNorm(a_i) = a_i / RMS(a) * g_i
    where RMS(a) = sqrt(1/d_model * sum(a_i^2) + eps)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.
        
        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        # Learnable gain parameter, initialized to 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor.
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Normalized tensor of the same shape
        """
        # Upcast to float32 for numerical stability
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Compute RMS
        # x: (..., d_model)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and apply gain
        x_normed = x / rms * self.weight.to(torch.float32)
        
        return x_normed.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU (Swish) activation function.
    SiLU(x) = x * sigmoid(x)
    
    Args:
        x: Input tensor of arbitrary shape
    
    Returns:
        Output tensor of the same shape
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network.
    
    FFN(x) = W2(SiLU(W1 @ x) * W3 @ x)
    
    where W1, W3: (d_ff, d_model) and W2: (d_model, d_ff)
    """
    
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Construct the SwiGLU feed-forward network.
        
        Args:
            d_model: Dimensionality of the input and output
            d_ff: Dimensionality of the inner feed-forward layer
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU transformation.
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Output tensor of shape (..., d_model)
        """
        # W1 @ x: (..., d_ff)
        # W3 @ x: (..., d_ff)
        # SiLU(W1 @ x) * (W3 @ x): (..., d_ff)
        # W2 @ (...): (..., d_model)
        return self.w2(silu(self.w1(x)) * self.w3(x))


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax to the specified dimension.
    
    softmax(x)_i = exp(x_i) / sum(exp(x_j))
    
    Uses the trick of subtracting the maximum for numerical stability.
    
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension to apply softmax to
    
    Returns:
        Output tensor with softmax applied to the specified dimension
    """
    # Subtract max for numerical stability
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    
    # Compute exp and normalize
    exp_x = torch.exp(x_shifted)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    return exp_x / sum_exp


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    
    For a query/key vector at position i, we apply pairwise rotations:
    R_i rotates pairs (q_{2k-1}, q_{2k}) by angle theta_{i,k} = i * theta^(2k/d)
    where theta is the base frequency parameter.
    """
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module.
        
        Args:
            theta: Base frequency (Θ) for RoPE
            d_k: Dimension of query and key vectors
            max_seq_len: Maximum sequence length
            device: Device to store the buffers on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Precompute frequency bands
        # theta_k = theta^(2(k-1)/d_k) for k in 1, ..., d_k/2
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        
        # Precompute position * frequency for all positions
        positions = torch.arange(max_seq_len, device=device).float()
        angles = torch.outer(positions, freqs)  # (max_seq_len, d_k/2)
        
        # Precompute cos and sin
        cos_cache = torch.cos(angles)  # (max_seq_len, d_k/2)
        sin_cache = torch.sin(angles)  # (max_seq_len, d_k/2)
        
        # Register as buffers (not parameters, not saved in state_dict by default)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to the input tensor.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Tensor of shape (..., seq_len) with token positions
        
        Returns:
            Tensor of shape (..., seq_len, d_k) with RoPE applied
        """
        # Get cos and sin for the given positions
        # token_positions: (..., seq_len)
        # cos_cache, sin_cache: (max_seq_len, d_k/2)
        
        # Index into the cache using token_positions
        # We need to gather cos and sin values for each position
        cos = self.cos_cache[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cache[token_positions]  # (..., seq_len, d_k/2)
        
        # Split x into pairs and apply rotation
        # x: (..., seq_len, d_k)
        x_reshape = rearrange(x, "... seq (d2 pair) -> ... seq d2 pair", pair=2)
        
        # Apply rotation to each pair
        # [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        x1 = x_reshape[..., 0]  # (..., seq_len, d_k/2)
        x2 = x_reshape[..., 1]  # (..., seq_len, d_k/2)
        
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        # Recombine
        x_rotated = torch.stack([x1_rot, x2_rot], dim=-1)
        x_out = rearrange(x_rotated, "... seq d2 pair -> ... seq (d2 pair)")
        
        return x_out


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q: Query tensor of shape (..., queries, d_k)
        K: Key tensor of shape (..., keys, d_k)
        V: Value tensor of shape (..., keys, d_v)
        mask: Optional boolean mask of shape (..., queries, keys)
              True means attend, False means don't attend
    
    Returns:
        Output tensor of shape (..., queries, d_v)
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores
    # Q: (..., queries, d_k)
    # K: (..., keys, d_k)
    # scores: (..., queries, keys)
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # Set masked positions to -inf
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax
    attn_weights = softmax(scores, dim=-1)
    
    # Apply attention to values
    # attn_weights: (..., queries, keys)
    # V: (..., keys, d_v)
    # output: (..., queries, d_v)
    output = einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    
    return output


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional RoPE.
    
    MultiHeadSelfAttention(x) = W_O @ MultiHead(W_Q @ x, W_K @ x, W_V @ x)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        theta: float = None,
        use_rope: bool = True,
        device=None,
        dtype=None
    ):
        """
        Construct multi-head self-attention.
        
        Args:
            d_model: Dimensionality of the input
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length (required if use_rope=True)
            theta: RoPE theta parameter (required if use_rope=True)
            use_rope: Whether to use RoPE
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        
        # Projection layers
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # RoPE
        if use_rope:
            assert max_seq_len is not None and theta is not None
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional tensor of shape (..., seq_len) with token positions
        
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        seq_len = x.shape[-2]
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (..., seq_len, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        # (..., seq_len, d_model) -> (..., seq_len, num_heads, d_k) -> (..., num_heads, seq_len, d_k)
        Q = rearrange(Q, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        K = rearrange(K, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        V = rearrange(V, "... seq (heads d_v) -> ... heads seq d_v", heads=self.num_heads)
        
        # Apply RoPE if enabled
        if self.use_rope and self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
                # Expand to match batch dimensions
                token_positions = token_positions.expand(Q.shape[:-2] + (seq_len,))
            
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        # Create causal mask
        # mask[i, j] = True if j <= i (can attend to past and current positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        mask = mask.unsqueeze(0)  # (1, seq_len, seq_len)
        
        # Apply scaled dot-product attention
        # Q, K, V: (..., num_heads, seq_len, d_k/d_v)
        # We need mask to broadcast over batch and head dimensions
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)
        
        # Reshape back
        # (..., num_heads, seq_len, d_v) -> (..., seq_len, num_heads, d_v) -> (..., seq_len, d_model)
        attn_output = rearrange(attn_output, "... heads seq d_v -> ... seq (heads d_v)")
        
        # Output projection
        output = self.output_proj(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block.
    
    y = x + MultiHeadSelfAttention(RMSNorm(x))
    y = y + FFN(RMSNorm(y))
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None
    ):
        """
        Construct a Transformer block.
        
        Args:
            d_model: Dimensionality of the input
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward inner layer
            max_seq_len: Maximum sequence length
            theta: RoPE theta parameter
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        # Pre-norm layers
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta, use_rope=True,
            device=device, dtype=dtype
        )
        
        # Feed-forward network
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply the Transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            token_positions: Optional tensor of shape (batch, seq_len) with token positions
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # First sublayer: attention with residual
        x = x + self.attn(self.ln1(x), token_positions)
        
        # Second sublayer: FFN with residual
        x = x + self.ffn(self.ln2(x))
        
        return x


class TransformerLM(nn.Module):
    """
    Transformer Language Model.
    
    Architecture:
    1. Token embeddings
    2. num_layers Transformer blocks
    3. Final RMSNorm
    4. Linear projection to vocabulary (LM head)
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None
    ):
        """
        Construct the Transformer LM.
        
        Args:
            vocab_size: Size of the vocabulary
            context_length: Maximum context length
            d_model: Dimensionality of the model
            num_layers: Number of Transformer layers
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward inner layer
            rope_theta: RoPE theta parameter
            device: Device to store parameters on
            dtype: Data type of parameters
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        
        # Token embeddings
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        
        # LM head (output projection)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer LM.
        
        Args:
            input_ids: Tensor of shape (batch, seq_len) with token IDs
        
        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        x = self.token_embeddings(input_ids)  # (batch, seq_len, d_model)
        
        # Create token positions
        token_positions = torch.arange(seq_len, device=input_ids.device)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)  # (batch, seq_len)
        
        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, token_positions)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        return logits
