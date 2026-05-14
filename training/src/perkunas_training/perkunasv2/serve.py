from __future__ import annotations

import argparse
import re
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from perkunas_training.perkunasv2.inference import (
    PerkunasV2ShardGenerator,
    result_to_dict,
)


class GenerateRequest(BaseModel):
    model: str = Field(default="primary")
    prompt: str
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    seed: int | None = None
    use_kv_cache: bool = True
    suppress_special_tokens: bool = True


class CompareRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=32, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    seed: int | None = None
    suppress_special_tokens: bool = True


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="primary")
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1, le=256)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    use_kv_cache: bool = True
    suppress_special_tokens: bool = True


class CompletionRequest(BaseModel):
    model: str = Field(default="primary")
    prompt: str
    max_tokens: int | None = Field(default=None, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    use_kv_cache: bool = True
    suppress_special_tokens: bool = True


def create_app(
    *,
    primary_run_dir: str | Path,
    backup_run_dir: str | Path,
    primary_tokenizer_dir: str | Path,
    backup_tokenizer_dir: str | Path,
    device: str,
    dtype: str,
    max_resident_shards: int,
    cache_active_modules: bool = False,
    preload_modules: bool = False,
) -> FastAPI:
    app = FastAPI(title="Perkunasv2 shard-native test host")
    primary = PerkunasV2ShardGenerator(
        primary_run_dir,
        tokenizer_dir=primary_tokenizer_dir,
        device=device,
        dtype=dtype,
        max_resident_shards=max_resident_shards,
        cache_active_modules=cache_active_modules,
        preload_modules=preload_modules,
    )
    same_backup = same_path(primary_run_dir, backup_run_dir) and same_path(
        primary_tokenizer_dir,
        backup_tokenizer_dir,
    )
    backup = (
        primary
        if same_backup
        else PerkunasV2ShardGenerator(
            backup_run_dir,
            tokenizer_dir=backup_tokenizer_dir,
            device=device,
            dtype=dtype,
            max_resident_shards=max_resident_shards,
            cache_active_modules=cache_active_modules,
            preload_modules=preload_modules,
        )
    )
    models = {"primary": primary, "backup": backup}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "models": sorted(models)}

    @app.get("/models")
    def list_models() -> dict[str, Any]:
        return {
            "models": [
                {
                    "id": name,
                    "run_dir": str(generator.run_dir),
                    "config": generator.config.to_dict(),
                    "residency": generator.store.residency_snapshot(),
                }
                for name, generator in models.items()
            ]
        }

    @app.get("/v1/models")
    def list_openai_models() -> dict[str, Any]:
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": created,
                    "owned_by": "perkunasv2-local",
                }
                for name in sorted(models)
            ],
        }

    @app.post("/v1/models/{model_name}/preload")
    def preload_openai_model(model_name: str) -> dict[str, Any]:
        generator = get_model(models, model_name)
        return {"model": model_name, **generator.preload_modules()}

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        generator = get_model(models, request.model)
        result = generator.generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed,
            model_name=request.model,
            use_kv_cache=request.use_kv_cache,
            suppress_special_tokens=request.suppress_special_tokens,
        )
        return result_to_dict(result)

    @app.post("/compare")
    def compare(request: CompareRequest) -> dict[str, Any]:
        outputs = []
        for name, generator in models.items():
            result = generator.generate(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            seed=request.seed,
            model_name=name,
            use_kv_cache=True,
            suppress_special_tokens=request.suppress_special_tokens,
        )
            outputs.append(result_to_dict(result))
        return {"prompt": request.prompt, "outputs": outputs}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not implemented yet")
        generator = get_model(models, request.model)
        prompt = render_chat_prompt(request.messages)
        prompt_token_count = len(generator.encode_generation_prompt(prompt))
        max_tokens = request.max_tokens or request.max_completion_tokens or 32
        result = generator.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed,
            model_name=request.model,
            stop_on_eos=True,
            use_kv_cache=request.use_kv_cache,
            suppress_special_tokens=request.suppress_special_tokens,
        )
        content, stop_matched = apply_stop_sequences(
            generator.tokenizer.decode(result.generated_token_ids),
            request.stop,
        )
        completion_tokens = len(result.generated_token_ids)
        finish_reason = finish_reason_for(result.generated_token_ids, generator.config.eos_token_id, stop_matched)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_token_count + completion_tokens,
            },
        }

    @app.post("/v1/completions")
    def completions(request: CompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="stream=true is not implemented yet")
        generator = get_model(models, request.model)
        max_tokens = request.max_tokens or 32
        prompt_token_count = len(generator.encode_generation_prompt(request.prompt))
        result = generator.generate(
            request.prompt,
            max_new_tokens=max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed,
            model_name=request.model,
            stop_on_eos=True,
            use_kv_cache=request.use_kv_cache,
            suppress_special_tokens=request.suppress_special_tokens,
        )
        text, stop_matched = apply_stop_sequences(
            generator.tokenizer.decode(result.generated_token_ids),
            request.stop,
        )
        completion_tokens = len(result.generated_token_ids)
        finish_reason = finish_reason_for(result.generated_token_ids, generator.config.eos_token_id, stop_matched)
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "text": text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_token_count + completion_tokens,
            },
        }

    return app


def get_model(models: dict[str, PerkunasV2ShardGenerator], name: str) -> PerkunasV2ShardGenerator:
    try:
        return models[name]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown model: {name}") from exc


def same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return Path(left) == Path(right)


def render_chat_prompt(messages: list[ChatMessage]) -> str:
    user_prompts: list[str] = []
    assistant_context: list[str] = []
    for message in messages:
        role = message.role.strip().lower() or "user"
        content = render_message_content(message.content).strip()
        if not content:
            continue
        if role == "assistant":
            assistant_context.append(content)
        elif role == "user":
            user_prompts.append(content)
    if assistant_context:
        prompt = "\n\n".join([*assistant_context, *user_prompts])
    elif user_prompts:
        prompt = user_prompts[-1]
    else:
        prompt = "\n\n".join(
            render_message_content(message.content).strip()
            for message in messages
            if render_message_content(message.content).strip()
        )
    return adapt_story_instruction_prompt(prompt)


def adapt_story_instruction_prompt(prompt: str) -> str:
    text = " ".join(prompt.split())
    if not text:
        return text
    lower = text.lower()
    prefixes = (
        "write a story about ",
        "write a tiny story about ",
        "tell me a story about ",
        "tell a story about ",
        "tell me about ",
    )
    subject = None
    for prefix in prefixes:
        if lower.startswith(prefix):
            subject = text[len(prefix) :].strip(" .")
            break
    if subject is None:
        return prompt

    named_match = re.search(
        r"\b(?:(?P<article>a|an|the)\s+)?(?P<kind>[A-Za-z]+)\s+named\s+(?P<name>[A-Za-z][A-Za-z0-9_-]*)",
        subject,
        flags=re.IGNORECASE,
    )
    if not named_match:
        return f"Once upon a time, there was {subject}."

    article = named_match.group("article") or article_for(named_match.group("kind"))
    kind = named_match.group("kind").lower()
    name = named_match.group("name")
    opening = f"Once upon a time, there was {article} {kind} named {name}."
    lost_match = re.search(
        r"\blost\s+(?:(?P<owner>his|her|their|the)\s+)?(?P<object>[A-Za-z0-9 -]+)$",
        subject,
        flags=re.IGNORECASE,
    )
    if lost_match:
        owner = (lost_match.group("owner") or "").lower()
        thing = lost_match.group("object").strip(" .")
        if thing:
            held = thing if owner in {"the", "their"} else f"{article_for(thing)} {thing}"
            lost = f"{owner} {thing}" if owner else f"{article_for(thing)} {thing}"
            return f"{opening} {name} had {held}. One day, {name} lost {lost}."
    return opening


def article_for(text: str) -> str:
    stripped = text.strip().lower()
    if not stripped:
        return "a"
    if stripped.split()[0] in {"a", "an", "the", "his", "her", "their", "my"}:
        return ""
    return "an" if stripped[0] in "aeiou" else "a"


def render_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def normalize_stop_sequences(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [item for item in stop if item]


def apply_stop_sequences(text: str, stop: str | list[str] | None) -> tuple[str, bool]:
    best_index: int | None = None
    for sequence in normalize_stop_sequences(stop):
        index = text.find(sequence)
        if index >= 0 and (best_index is None or index < best_index):
            best_index = index
    if best_index is None:
        return text, False
    return text[:best_index], True


def finish_reason_for(
    generated_token_ids: list[int],
    eos_token_id: int,
    stop_matched: bool,
) -> str:
    if stop_matched:
        return "stop"
    if generated_token_ids and generated_token_ids[-1] == eos_token_id:
        return "stop"
    return "length"


def main() -> None:
    parser = argparse.ArgumentParser(description="Host primary and backup Perkunasv2 runs")
    parser.add_argument("--primary-run-dir", required=True)
    parser.add_argument("--backup-run-dir", required=True)
    parser.add_argument("--primary-tokenizer-dir", required=True)
    parser.add_argument("--backup-tokenizer-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--max-resident-shards", type=int, default=1)
    parser.add_argument("--cache-active-modules", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preload-modules", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    import uvicorn

    app = create_app(
        primary_run_dir=args.primary_run_dir,
        backup_run_dir=args.backup_run_dir,
        primary_tokenizer_dir=args.primary_tokenizer_dir,
        backup_tokenizer_dir=args.backup_tokenizer_dir,
        device=args.device,
        dtype=args.dtype,
        max_resident_shards=args.max_resident_shards,
        cache_active_modules=args.cache_active_modules,
        preload_modules=args.preload_modules,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
