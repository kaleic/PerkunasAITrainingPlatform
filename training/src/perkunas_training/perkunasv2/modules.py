from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from perkunas_training.perkunasv2.configuration import PerkunasV2Config


class ShardRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x_norm).to(dtype)


class ShardRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        end = position_offset + seq_len
        cos = self.cos_cached[:, :, position_offset:end, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, position_offset:end, :].to(dtype=q.dtype, device=q.device)
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class ShardSelfAttention(nn.Module):
    def __init__(self, config: PerkunasV2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = ShardRotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(y)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        *,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k, position_offset=position_offset)
        if past_key is None:
            key = k
            value = v
            y = F.scaled_dot_product_attention(q, key, value, is_causal=True)
        else:
            if past_value is None:
                raise ValueError("past_value is required when past_key is provided")
            key = torch.cat((past_key, k), dim=-2)
            value = torch.cat((past_value, v), dim=-2)
            if seq_len == 1:
                y = F.scaled_dot_product_attention(q, key, value, is_causal=False)
            else:
                key_positions = torch.arange(key.shape[-2], device=x.device)
                query_positions = torch.arange(
                    position_offset,
                    position_offset + seq_len,
                    device=x.device,
                )
                mask = key_positions[None, :] <= query_positions[:, None]
                y = F.scaled_dot_product_attention(q, key, value, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(y), key, value


class ShardSwiGLU(nn.Module):
    def __init__(self, config: PerkunasV2Config):
        super().__init__()
        self.gate_up = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class ShardTransformerBlock(nn.Module):
    def __init__(self, config: PerkunasV2Config, block_index: int):
        super().__init__()
        self.block_index = block_index
        self.input_layernorm = ShardRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = ShardSelfAttention(config)
        self.post_attention_layernorm = ShardRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = ShardSwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        *,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_output, key, value = self.self_attn.forward_with_cache(
            self.input_layernorm(x),
            past_key=past_key,
            past_value=past_value,
            position_offset=position_offset,
        )
        x = x + attn_output
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, key, value


def build_embeddings(config: PerkunasV2Config) -> nn.Module:
    return nn.Embedding(config.vocab_size, config.hidden_size)


def build_transformer_block(config: PerkunasV2Config, block_index: int) -> nn.Module:
    return ShardTransformerBlock(config, block_index)


def build_final_norm(config: PerkunasV2Config) -> nn.Module:
    return ShardRMSNorm(config.hidden_size, config.rms_norm_eps)


def build_lm_head(config: PerkunasV2Config) -> nn.Module:
    return nn.Linear(config.hidden_size, config.vocab_size, bias=False)


def initialize_module(module: nn.Module, config: PerkunasV2Config) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.normal_(child.weight, mean=0.0, std=config.initializer_range)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Embedding):
            nn.init.normal_(child.weight, mean=0.0, std=config.initializer_range)
