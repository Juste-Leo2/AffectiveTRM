# layers.py

from typing import Tuple
import einops
import torch
from torch import nn
import torch.nn.functional as F
import math

# --- Fonction 'trunc_normal_init_' copiée ici pour l'autonomie du fichier ---
def trunc_normal_init_(tensor: torch.Tensor, mean: float = 0, std: float = 1.0) -> torch.Tensor:
    """Truncated normal initialization."""
    # Get upper and lower bounds
    # Values more than 2 std from the mean are recreated
    low = -2 * std
    high = 2 * std

    with torch.no_grad():
        # Repeatedly sample from the normal distribution until all values are within the desired range
        torch.nn.init.normal_(tensor, mean=mean, std=std)
        while torch.any(tensor < low) or torch.any(tensor > high):
            # Find the indices of the values that are out of bounds
            out_of_bounds = (tensor < low) | (tensor > high)
            # Resample the out-of-bounds values
            tensor[out_of_bounds] = torch.normal(
                mean, std, size=(torch.sum(out_of_bounds),), device=tensor.device
            )
    return tensor
# -------------------------------------------------------------------------

#try:
#    from flash_attn_interface import flash_attn_func  # type: ignore[import]
#except ImportError:
#    # Fallback to FlashAttention 2
#    from flash_attn import flash_attn_func  # type: ignore[import]
from torch.nn.functional import scaled_dot_product_attention


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        # On ignore num_key_value_heads pour l'instant car on utilise une MHA standard
        # où num_heads_q = num_heads_k = num_heads_v

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.output_size = head_dim * num_heads
        self.causal = causal

        # Projection 3 x D pour Q, K, V
        self.qkv_proj = CastedLinear(self.hidden_size, 3 * self.output_size, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Calculer Q, K, V en une seule fois
        qkv = self.qkv_proj(hidden_states)
        
        # Séparer Q, K, V
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            # Attention, on a une séquence de longueur 1
            # Il faut s'assurer que cos/sin ont la bonne dimension
            # Le rotary_emb a été créé avec max_position_embeddings=2
            # On prend la première position pour notre séquence de longueur 1
            cos = cos[:seq_len]
            sin = sin[:seq_len]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Transposer pour l'attention: (B, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # scaled_dot_product_attention
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        
        # Recombiner les têtes
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.output_size)
        
        return self.o_proj(attn_output)



class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)