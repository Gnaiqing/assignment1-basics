import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum, reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def initialize_linear_weight(out_features, in_features, device=None, dtype=None):
    """
    Helper function that initialize linear weight matrix
    """
    sigma = (2 / (in_features + out_features)) ** 0.5
    weight = nn.init.trunc_normal_(torch.empty((out_features, in_features), device=device, dtype=dtype),
                                   std=sigma, a=-3 * sigma, b=3 * sigma)
    return nn.Parameter(weight)


class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super(Linear, self).__init__()
        self.weight = initialize_linear_weight(out_features, in_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = einsum(
            x, self.weight,
            "... d_in, d_out d_in -> ... d_out"
        )
        return output


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super(Embedding, self).__init__()
        embedding = nn.init.trunc_normal_(torch.empty((num_embeddings, embedding_dim),
                                                      device=device, dtype=dtype), std=1, a=-3, b=3)
        self.embedding_table = nn.Parameter(embedding)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_table[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm = torch.sqrt(torch.sum(torch.square(x), dim=-1, keepdim=True)/self.d_model + self.eps)
        result = x * self.gain / norm
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.W1 = initialize_linear_weight(d_ff, d_model, device=device, dtype=dtype)
        self.W2 = initialize_linear_weight(d_model, d_ff, device=device,dtype=dtype)
        self.W3 = initialize_linear_weight(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        l1 = F.sigmoid(einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff"))
        silu = l1 * einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        l2 = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        out = einsum(silu * l2, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return out


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE; got {d_k}")
        half = d_k // 2
        i = torch.arange(0, half, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (2 * i / d_k))
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # freqs: [max_seq_len, half]
        freqs = einsum(pos, inv_freq, " n, d -> n d")
        self.register_buffer('sin', torch.sin(freqs), persistent=False)
        self.register_buffer('cos', torch.cos(freqs), persistent=False)
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        x:               (..., seq_len, d_k)
        token_positions: (..., seq_len)  (long/int)
        Returns:         (..., seq_len, d_k)
        """
        if x.size(-1) != self.d_k:
            raise ValueError(f"Last dim of x must be d_k={self.d_k}; got {x.size(-1)}")

        # Ensure positions are long and in range
        pos = token_positions.long()
        if pos.max().item() >= self.max_seq_len or (pos.min().item() < 0):
            raise ValueError(f"token_positions out of range [0, {self.max_seq_len - 1}]")

        # Gather sin/cos rows for these positions -> (..., seq_len, d_k/2)
        # Advanced indexing works with multi-dim index on first dim.
        sin = self.sin[pos].to(dtype=x.dtype, device=x.device)
        cos = self.cos[pos].to(dtype=x.dtype, device=x.device)

        # Split last dim into pairs (real, imag) for rotation
        # x_even = x[..., 0::2], x_odd = x[..., 1::2]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply complex rotation:
        # [x_even, x_odd] rotated by angle = freqs:
        # x_even' = x_even * cos - x_odd * sin
        # x_odd'  = x_odd  * cos + x_even * sin
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_odd * cos + x_even * sin

        # Interleave back to shape (..., seq_len, d_k)
        # Stack along the last new dim of size 2 and reshape
        y = torch.empty_like(x)
        y[..., 0::2] = x_even_rot
        y[..., 1::2] = x_odd_rot
        return y


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Get the softmax of x through dimension i
    """
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    return torch.exp(x - x_max) / torch.sum(torch.exp(x - x_max), dim=dim, keepdim=True)


def scaled_dot_product_attention(query: Float[Tensor, "b ... n d_k"],
                                 key: Float[Tensor, "b ... m d_k"],
                                 value: Float[Tensor, "b ... m d_v"],
                                 mask: Bool[Tensor, "n m"] | None = None):
    """
    dot product attention with optional mask
    return an output with the shape (batch_size,..., n, d_v).
    """
    d_k = query.size(dim=-1)
    dot_product = einsum(query, key, "b ... n d_k , b ... m d_k -> b ... n m") / (d_k ** 0.5)
    if mask is not None:
        dot_product.masked_fill_(~mask, -torch.inf)

    softmax_output = softmax(dot_product, dim=-1)
    attention = einsum(softmax_output, value, "b ... n m , b ... m d_v -> b ... n d_v")
    return attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        d_k = d_model // num_heads
        self.WQ = initialize_linear_weight(num_heads * d_k, d_model, device=device, dtype=dtype)
        self.WK = initialize_linear_weight(num_heads * d_k, d_model, device=device, dtype=dtype)
        self.WV = initialize_linear_weight(num_heads * d_k, d_model, device=device, dtype=dtype)
        self.WO = initialize_linear_weight(d_model, num_heads * d_k, device=device, dtype=dtype)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=device) if theta else None

    def forward(self, x: Float[Tensor, "b ... seq_len d_model"],
                token_positions: Int[Tensor, " ... seq_len"] | None = None) -> Float[Tensor, "b ... seq_len d_model"]:
        # project
        WQ_re = rearrange(self.WQ, "(h d_k) d_model -> h d_k d_model", d_k=self.d_k)
        WK_re = rearrange(self.WK, "(h d_k) d_model -> h d_k d_model", d_k=self.d_k)
        WV_re = rearrange(self.WV, "(h d_k) d_model -> h d_k d_model", d_k=self.d_k)
        WO_re = rearrange(self.WO, "d_model (h d_k) -> d_model h d_k", d_k=self.d_k)

        q = einsum(x, WQ_re, "b ... n d_model, h d_k d_model -> b h ... n d_k").contiguous()
        k = einsum(x, WK_re, "b ... n d_model, h d_k d_model -> b h ... n d_k").contiguous()
        v = einsum(x, WV_re, "b ... n d_model, h d_k d_model -> b h ... n d_k").contiguous()

        if self.rope:
            if token_positions is None:
                seq_len = q.shape[-2]
                pos = torch.arange(seq_len, device=x.device)
                token_positions = pos.expand(*q.shape[0:2], seq_len) if q.dim() == 4 else pos
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Flash/SDPA — no explicit mask allocation; use is_causal=True
        # Shapes expected: (B, H, L, D)
        B, H = q.shape[0], q.shape[1]
        q = q.view(B, H, -1, self.d_k)
        k = k.view(B, H, -1, self.d_k)
        v = v.view(B, H, -1, self.d_k)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        # back to (B, ..., L, D) then project out
        attn = attn.view_as(q)
        out = einsum(attn, WO_re, "b h ... n d_k, d_model h d_k -> b ... n d_model")
        return out


# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, d_model: int, num_heads: int, max_seq_len=None, theta=None, device=None, dtype=None):
#         super(MultiHeadSelfAttention, self).__init__()
#         d_k = d_model // num_heads
#         self.WQ = initialize_linear_weight(num_heads * d_k, d_model, device=device, dtype=dtype)
#         self.WK = initialize_linear_weight(num_heads * d_k, d_model, device=device, dtype=dtype)
#         self.WV = initialize_linear_weight(num_heads * d_k, d_model, device=device, dtype=dtype)
#         self.WO = initialize_linear_weight(d_model, num_heads * d_k, device=device, dtype=dtype)
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_k
#         if theta:
#             self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=device)
#         else:
#             self.rope = None
#
#     def forward(self, x: Float[Tensor, "b ... seq_len d_model"],
#                 token_positions: Int[Tensor, " ... seq_len"] | None = None) -> Float[Tensor, "b ... seq_len d_model"]:
#         """
#         Multihead self attention on input x
#         """
#         WQ_re = rearrange(self.WQ, "(h d_k) d_model -> h d_k d_model", d_k=self.d_k)
#         WK_re = rearrange(self.WK, "(h d_k) d_model -> h d_k d_model", d_k=self.d_k)
#         WV_re = rearrange(self.WV, "(h d_k) d_model -> h d_k d_model", d_k=self.d_k)
#         WO_re = rearrange(self.WO, "d_model (h d_k) -> d_model h d_k", d_k=self.d_k)
#         q_vec = einsum(x, WQ_re, "b ... seq_len d_model, h d_k d_model -> b h ... seq_len d_k")
#         k_vec = einsum(x, WK_re, "b ... seq_len d_model, h d_k d_model -> b h ... seq_len d_k")
#         v_vec = einsum(x, WV_re, "b ... seq_len d_model, h d_k d_model -> b h ... seq_len d_k")
#         if self.rope:
#             if token_positions is None:
#                 *batch_and_prefix, seq_len, _ = q_vec.shape
#                 target_shape = (*batch_and_prefix, seq_len)
#                 device = x.device
#                 # same positions for all batch items
#                 token_positions = torch.arange(seq_len, device=device)  # (seq_len)
#                 # broadcast to batch (and other prefix dims)
#                 token_positions = token_positions.expand(target_shape)
#
#             q_vec = self.rope(x=q_vec, token_positions=token_positions)
#             k_vec = self.rope(x=k_vec, token_positions=token_positions)
#
#         seq_len = x.size(dim=-2)
#         mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)).T
#         attention_vec = scaled_dot_product_attention(q_vec, k_vec, v_vec, mask) # "b h ... seq_len d_k"
#         output = einsum(attention_vec, WO_re, "b h ... seq_len d_k, d_model h d_k -> b ... seq_len d_model")
#         return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len=None, theta=None, device=None, dtype=None):
        super(TransformerBlock, self).__init__()
        self.attention_block = nn.Sequential(
            RMSNorm(d_model, device=device, dtype=dtype),
            MultiHeadSelfAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta,
                                   device=device, dtype=dtype)
        )
        self.ff_block = nn.Sequential(
            RMSNorm(d_model, device=device, dtype=dtype),
            SwiGLU(d_model, d_ff, device=device, dtype=dtype),
        )

    def forward(self, x: Float[Tensor, "b ... seq_len d_model"]):
        x = x + self.attention_block(x)
        x = x + self.ff_block(x)
        return x


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device=None,
                 dtype=None):
        """
        Initialize transformer model
        """
        super(Transformer, self).__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta,
                                                device=device, dtype=dtype))
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, "batch_size sequence_length"]) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
        """
        Feed in batch of text and predict unnormalized next token
        """
        x = self.token_embeddings(x)
        # x = self.layers(x)
        for block in self.layers:
            # checkpoint wants a function that consumes tensors only
            def fn(inp):
                return block(inp)

            x = checkpoint(fn, x, use_reentrant=False)
        x = self.lm_head(self.ln_final(x))
        return x









