from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from kvserve.api.schemas import ChatCompletionRequest
from kvserve.backends.base import EmbeddingResult, GenerationChunk, GenerationResult, RerankResult
from kvserve.models.schemas import ModelSpec
from kvserve.quantization.applier import build_transformers_quantization_config


class TransformersBackend:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers backend requires torch and transformers") from exc

        model_name = spec.backend_config.get("model_name_or_path", spec.model_id)
        quantization_config = build_transformers_quantization_config(spec)
        dtype = torch.bfloat16 if spec.quantization_mode == "bf16" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=spec.backend_config.get("device_map", "auto"),
            torch_dtype=dtype,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        self.model.eval()

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def apply_chat_template(self, messages: list[dict[str, Any]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        rendered = []
        for message in messages:
            rendered.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        rendered.append("assistant:")
        return "\n".join(rendered)

    async def generate(self, prompt: str, request: ChatCompletionRequest) -> GenerationResult:
        import torch

        max_new_tokens = request.output_token_limit
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = await asyncio.to_thread(
                self.model.generate,
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_tokens=int(inputs["input_ids"].numel()),
            completion_tokens=int(generated.numel()),
            finish_reason="stop",
        )

    async def generate_stream(
        self, prompt: str, request: ChatCompletionRequest
    ) -> AsyncIterator[GenerationChunk]:
        result = await self.generate(prompt, request)
        for token in result.text.split(" "):
            yield GenerationChunk(text=token + " ", token_count=1)
            await asyncio.sleep(0)
        yield GenerationChunk(text="", token_count=0, finish_reason=result.finish_reason)

    async def embed(self, inputs: list[str], dimensions: int | None = None) -> EmbeddingResult:
        raise RuntimeError("use a dedicated embedding backend/model for embeddings")

    async def rerank(self, query: str, documents: list[str]) -> RerankResult:
        raise RuntimeError("use a dedicated reranking backend/model for reranking")
