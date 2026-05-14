from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator

import torch
from tokenizers import Tokenizer

from kvserve.api.schemas import ChatCompletionRequest
from kvserve.backends.base import EmbeddingResult, GenerationChunk, GenerationResult, RerankResult
from kvserve.models.schemas import ModelSpec
from perkunas_training.model.modeling_perkunas import PerkunasForCausalLM


class PerkunasBackend:
    """Direct PyTorch backend for Perkunas checkpoints exported by training/export_hf.py."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        model_path = Path(spec.backend_config["model_name_or_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Perkunas artifact path not found: {model_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("Perkunas backend requires CUDA; refusing CPU serving")
        self.device = torch.device(spec.backend_config.get("device", "cuda:0"))
        torch.cuda.set_device(self.device)
        self.tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
        self.model = PerkunasForCausalLM.from_pretrained(model_path, map_location="cpu")
        self.model.to(self.device)
        self.model.eval()
        if next(self.model.parameters()).device.type != "cuda":
            raise RuntimeError("Perkunas model failed to move to CUDA")
        self.max_context = int(spec.backend_config.get("max_context", self.model.config.max_position_embeddings))
        self.default_top_k = int(spec.backend_config.get("top_k", 50))
        print(
            "Perkunas backend loaded on "
            f"{next(self.model.parameters()).device}; "
            f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB",
            flush=True,
        )

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    def apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        rendered: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content")
            if isinstance(content, list):
                text = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
            else:
                text = "" if content is None else str(content)
            rendered.append(f"{role}: {text}")
        rendered.append("assistant:")
        return "\n".join(rendered)

    async def generate(self, prompt: str, request: ChatCompletionRequest) -> GenerationResult:
        return await asyncio.to_thread(self._generate_sync, prompt, request)

    async def generate_stream(
        self, prompt: str, request: ChatCompletionRequest
    ) -> AsyncIterator[GenerationChunk]:
        result = await self.generate(prompt, request)
        for token in result.text.split(" "):
            if token:
                yield GenerationChunk(text=token + " ", token_count=1)
                await asyncio.sleep(0)
        yield GenerationChunk(text="", token_count=0, finish_reason=result.finish_reason)

    async def embed(self, inputs: list[str], dimensions: int | None = None) -> EmbeddingResult:
        raise RuntimeError("Perkunas generation backend does not serve embeddings")

    async def rerank(self, query: str, documents: list[str]) -> RerankResult:
        raise RuntimeError("Perkunas generation backend does not serve reranking")

    @torch.inference_mode()
    def _generate_sync(self, prompt: str, request: ChatCompletionRequest) -> GenerationResult:
        encoded = self.tokenizer.encode(prompt)
        prompt_ids = encoded.ids[-self.max_context :]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=request.output_token_limit,
            temperature=max(request.temperature, 1e-5),
            top_k=self.default_top_k,
        )
        completion_ids = output_ids[0, input_ids.shape[1] :].tolist()
        text = self.tokenizer.decode(completion_ids)
        return GenerationResult(
            text=text,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(completion_ids),
            finish_reason="length" if len(completion_ids) >= request.output_token_limit else "stop",
        )
