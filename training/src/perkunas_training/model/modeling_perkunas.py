from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from perkunas_training.model.configuration import PerkunasConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos = self.cos_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: PerkunasConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.dropout = config.attention_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k)
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.config.hidden_size)
        return self.o_proj(y)


class SwiGLUMLP(nn.Module):
    def __init__(self, config: PerkunasConfig):
        super().__init__()
        self.gate_up = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class PerkunasBlock(nn.Module):
    def __init__(self, config: PerkunasConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.resid_dropout(self.self_attn(self.input_layernorm(x)))
        x = x + self.resid_dropout(self.mlp(self.post_attention_layernorm(x)))
        return x


class PerkunasForCausalLM(nn.Module):
    def __init__(self, config: PerkunasConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([PerkunasBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.modality_encoders = nn.ModuleDict()
        self.modality_projectors = nn.ModuleDict()
        self.auxiliary_heads = nn.ModuleDict()
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def register_modality_encoder(
        self,
        name: str,
        encoder: nn.Module,
        encoder_output_dim: int,
        trainable_base: bool = False,
    ) -> None:
        self.modality_encoders[name] = encoder
        self.modality_projectors[name] = nn.Linear(encoder_output_dim, self.config.hidden_size, bias=False)
        if not trainable_base:
            for parameter in self.parameters():
                parameter.requires_grad = False
            for parameter in self.modality_encoders[name].parameters():
                parameter.requires_grad = True
            for parameter in self.modality_projectors[name].parameters():
                parameter.requires_grad = True

    def register_auxiliary_head(self, name: str, head: nn.Module, trainable_base: bool = False) -> None:
        self.auxiliary_heads[name] = head
        if not trainable_base:
            for parameter in self.parameters():
                parameter.requires_grad = False
            for parameter in self.auxiliary_heads[name].parameters():
                parameter.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds
        if x.shape[1] > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )
        for layer in self.layers:
            x = layer(x)
        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        if return_hidden_states:
            out["hidden_states"] = hidden
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_position_embeddings :]
            logits = self(context)["logits"][:, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits = torch.where(logits < values[:, [-1]], torch.full_like(logits, -math.inf), logits)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_json(output_dir / "config.json")
        save_model(self, str(output_dir / "model.safetensors"))

    @classmethod
    def from_pretrained(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "PerkunasForCausalLM":
        path = Path(path)
        config = PerkunasConfig.from_json(path / "config.json")
        model = cls(config)
        load_model(model, str(path / "model.safetensors"), device=str(map_location))
        return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
