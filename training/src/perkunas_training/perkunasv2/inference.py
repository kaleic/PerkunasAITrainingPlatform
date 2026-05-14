from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from perkunas_training.perkunasv2.configuration import PerkunasV2Config
from perkunas_training.perkunasv2.shard_store import ParameterShardStore, shard_names
from perkunas_training.perkunasv2.trainer import dtype_for_device, select_shard_device


@dataclass(slots=True)
class GenerationResult:
    model: str
    prompt: str
    text: str
    generated_token_ids: list[int]
    elapsed_seconds: float


class PerkunasV2ShardGenerator:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        tokenizer_dir: str | Path,
        device: str = "cuda",
        dtype: str = "fp16",
        max_resident_shards: int = 1,
        lm_head_chunk_tokens: int = 4096,
        cache_active_modules: bool = False,
        preload_modules: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.config = PerkunasV2Config.from_json(self.run_dir / "config.json")
        self.device = select_shard_device(device)
        self.dtype = dtype_for_device(dtype, self.device)
        self.store = ParameterShardStore(
            self.run_dir,
            config=self.config,
            max_resident_shards=max_resident_shards,
            clear_cuda_cache_between_shards=False,
            cache_active_modules=cache_active_modules,
        )
        tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(tokenizer_path)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.lm_head_chunk_tokens = lm_head_chunk_tokens
        self.cache_active_modules = cache_active_modules
        if preload_modules:
            self.preload_modules()

    @torch.no_grad()
    def preload_modules(self) -> dict[str, Any]:
        start = time.perf_counter()
        loaded: list[str] = []
        for shard_name in shard_names(self.config):
            with self.store.active_module(
                shard_name,
                device=self.device,
                dtype=self.dtype,
                training=False,
            ):
                loaded.append(shard_name)
        elapsed = time.perf_counter() - start
        residency = self.store.residency_snapshot()
        return {
            "loaded_shards": loaded,
            "elapsed_seconds": elapsed,
            "cache_active_modules": self.cache_active_modules,
            "residency": residency,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        seed: int | None = None,
        stop_on_eos: bool = True,
        model_name: str = "perkunasv2",
        use_kv_cache: bool = True,
        suppress_special_tokens: bool = True,
    ) -> GenerationResult:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        start = time.perf_counter()
        ids = self.encode_generation_prompt(prompt)
        original_len = len(ids)
        suppressed_token_ids = (
            self.generation_suppressed_token_ids()
            if suppress_special_tokens
            else []
        )
        if use_kv_cache:
            generated = self._generate_with_kv_cache(
                ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_on_eos=stop_on_eos,
                suppress_token_ids=suppressed_token_ids,
            )
            ids.extend(generated)
        else:
            for _ in range(max_new_tokens):
                context = ids[-self.config.max_position_embeddings :]
                input_ids = torch.tensor([context], dtype=torch.long)
                logits = self.forward_last_logits(input_ids)
                next_id = sample_next_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    suppress_token_ids=suppressed_token_ids,
                )
                ids.append(next_id)
                if stop_on_eos and next_id == self.config.eos_token_id:
                    break
        elapsed = time.perf_counter() - start
        return GenerationResult(
            model=model_name,
            prompt=prompt,
            text=self.tokenizer.decode(ids),
            generated_token_ids=ids[original_len:],
            elapsed_seconds=elapsed,
        )

    def encode_generation_prompt(self, prompt: str) -> list[int]:
        ids = self.tokenizer.encode(prompt).ids
        while ids and ids[-1] == self.config.eos_token_id:
            ids.pop()
        if not ids:
            ids = [self.config.bos_token_id]
        return ids

    def generation_suppressed_token_ids(self) -> list[int]:
        suppressed = {self.config.pad_token_id, self.config.bos_token_id}
        return sorted(token_id for token_id in suppressed if token_id is not None)

    @torch.no_grad()
    def _generate_with_kv_cache(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        stop_on_eos: bool,
        suppress_token_ids: Sequence[int],
    ) -> list[int]:
        if max_new_tokens <= 0:
            return []
        context = prompt_ids[-self.config.max_position_embeddings :]
        if len(context) >= self.config.max_position_embeddings:
            # No room to append cached positions; use the sliding-context path for this edge case.
            ids = list(prompt_ids)
            generated: list[int] = []
            for _ in range(max_new_tokens):
                input_ids = torch.tensor(
                    [ids[-self.config.max_position_embeddings :]],
                    dtype=torch.long,
                )
                logits = self.forward_last_logits(input_ids)
                next_id = sample_next_token(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    suppress_token_ids=suppress_token_ids,
                )
                ids.append(next_id)
                generated.append(next_id)
                if stop_on_eos and next_id == self.config.eos_token_id:
                    break
            return generated

        input_ids = torch.tensor([context], dtype=torch.long)
        logits, cache = self.forward_prefill_logits_with_cache(input_ids)
        generated = []
        position = len(context)
        for index in range(max_new_tokens):
            next_id = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                suppress_token_ids=suppress_token_ids,
            )
            generated.append(next_id)
            if stop_on_eos and next_id == self.config.eos_token_id:
                break
            if index == max_new_tokens - 1:
                break
            if position >= self.config.max_position_embeddings:
                break
            logits, cache = self.forward_next_logits_with_cache(next_id, cache, position)
            position += 1
        return generated

    @torch.no_grad()
    def forward_prefill_logits_with_cache(
        self,
        input_ids_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if input_ids_cpu.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")
        input_ids = input_ids_cpu.to(self.device, dtype=torch.long)
        with self.store.active_module(
            "embeddings", device=self.device, dtype=self.dtype, training=False
        ) as module:
            x = module(input_ids)
        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block_index in range(self.config.num_layers):
            shard_name = f"block_{block_index:03d}"
            with self.store.active_module(
                shard_name, device=self.device, dtype=self.dtype, training=False
            ) as module:
                x, key, value = transformer_block_forward_with_cache(
                    module,
                    x,
                    position_offset=0,
                )
            cache.append((key, value))
        logits = self._last_logits_from_hidden(x)
        return logits, cache

    @torch.no_grad()
    def forward_next_logits_with_cache(
        self,
        token_id: int,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        position_offset: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        with self.store.active_module(
            "embeddings", device=self.device, dtype=self.dtype, training=False
        ) as module:
            x = module(input_ids)
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block_index, (past_key, past_value) in enumerate(cache):
            shard_name = f"block_{block_index:03d}"
            with self.store.active_module(
                shard_name, device=self.device, dtype=self.dtype, training=False
            ) as module:
                x, key, value = transformer_block_forward_with_cache(
                    module,
                    x,
                    past_key=past_key,
                    past_value=past_value,
                    position_offset=position_offset,
                )
            next_cache.append((key, value))
        logits = self._last_logits_from_hidden(x)
        return logits, next_cache

    @torch.no_grad()
    def forward_last_logits(self, input_ids_cpu: torch.Tensor) -> torch.Tensor:
        if input_ids_cpu.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")
        input_ids = input_ids_cpu.to(self.device, dtype=torch.long)
        with self.store.active_module(
            "embeddings", device=self.device, dtype=self.dtype, training=False
        ) as module:
            x = module(input_ids)
        for block_index in range(self.config.num_layers):
            shard_name = f"block_{block_index:03d}"
            with self.store.active_module(
                shard_name, device=self.device, dtype=self.dtype, training=False
            ) as module:
                x = module(x)
        logits = self._last_logits_from_hidden(x)
        if not torch.isfinite(logits).all():
            raise RuntimeError("model produced non-finite logits")
        return logits.detach().cpu()

    @torch.no_grad()
    def _last_logits_from_hidden(self, x: torch.Tensor) -> torch.Tensor:
        with self.store.active_module(
            "final_norm", device=self.device, dtype=self.dtype, training=False
        ) as module:
            x = module(x)
        last_hidden = x[:, -1:, :]
        with self.store.active_module(
            "lm_head", device=self.device, dtype=self.dtype, training=False
        ) as module:
            logits = module(last_hidden).float()[0, -1, :]
        if not torch.isfinite(logits).all():
            raise RuntimeError("model produced non-finite logits")
        return logits.detach().cpu()


def transformer_block_forward_with_cache(
    module: nn.Module,
    x: torch.Tensor,
    *,
    past_key: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_with_cache = getattr(module, "forward_with_cache", None)
    if forward_with_cache is None:
        raise TypeError(f"{type(module).__name__} does not support KV-cache inference")
    return forward_with_cache(
        x,
        past_key=past_key,
        past_value=past_value,
        position_offset=position_offset,
    )


def sample_next_token(
    logits_cpu: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    suppress_token_ids: Sequence[int] | None = None,
) -> int:
    logits = logits_cpu.float()
    if suppress_token_ids:
        for token_id in suppress_token_ids:
            if 0 <= token_id < logits.numel():
                logits[token_id] = float("-inf")
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / max(1e-6, temperature)
    if top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, float("-inf"))
        filtered[indices] = values
        logits = filtered
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[0] = False
        sorted_logits[remove] = float("-inf")
        filtered = torch.full_like(logits, float("-inf"))
        filtered[sorted_indices] = sorted_logits
        logits = filtered
    probs = F.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
        return int(torch.argmax(logits_cpu).item())
    return int(torch.multinomial(probs, num_samples=1).item())


def result_to_dict(result: GenerationResult) -> dict[str, Any]:
    return {
        "model": result.model,
        "prompt": result.prompt,
        "text": result.text,
        "generated_token_ids": result.generated_token_ids,
        "elapsed_seconds": result.elapsed_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a Perkunasv2 shard-native run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--max-resident-shards", type=int, default=1)
    parser.add_argument("--cache-active-modules", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preload-modules", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kv-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suppress-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-name", default="perkunasv2")
    args = parser.parse_args()

    generator = PerkunasV2ShardGenerator(
        args.run_dir,
        tokenizer_dir=args.tokenizer_dir,
        device=args.device,
        dtype=args.dtype,
        max_resident_shards=args.max_resident_shards,
        cache_active_modules=args.cache_active_modules,
        preload_modules=args.preload_modules,
    )
    result = generator.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        model_name=args.model_name,
        use_kv_cache=args.kv_cache,
        suppress_special_tokens=args.suppress_special_tokens,
    )
    print(json.dumps(result_to_dict(result), indent=2))


if __name__ == "__main__":
    main()
