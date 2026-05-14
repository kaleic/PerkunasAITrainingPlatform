# Training and Weights

## Loading Pretrained Generation Models

1. Install GPU dependencies:

```powershell
pip install -e ".[gpu]"
```

2. Add a model to `config/model_registry.json`:

```json
{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "task_type": "generate",
  "backend": "vllm",
  "backend_config": {
    "model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
    "engine_args": {
      "gpu_memory_utilization": 0.90,
      "max_model_len": 32768
    }
  },
  "quantization_mode": "auto",
  "kv_compression_mode": "fp8",
  "kv_required": true,
  "max_context": 32768,
  "streaming_supported": true,
  "chat_template_required": true,
  "hardware_constraints": {
    "min_gpu_memory_gb": 24,
    "supports_fp8_kv": true,
    "requires_cuda": true
  },
  "policy_mode": "balanced"
}
```

3. Start:

```powershell
$env:KV_API_TOKENS="tenant_a:replace-me"
uvicorn kvserve.app:create_app --factory --host 0.0.0.0 --port 8000
```

## Applying Quantization

Pre-quantized models should set:

```json
"backend_config": {
  "prequantized": true,
  "model_name_or_path": "path-or-repo"
}
```

Online quantization is selected by `quantization_mode=auto` plus policy mode.
INT8 and INT4 Transformers loading uses bitsandbytes. vLLM loading maps the
registry mode into vLLM engine arguments.

## Tokenizer and Chat Templates

Generation models should set `chat_template_required=true`. Backends use the
tokenizer's native `apply_chat_template` when available. If a tokenizer has no
template, register a tokenizer revision with a chat template before production
promotion.

## Embedding Models

Register embedding models with:

```json
{
  "model_id": "your-embedding-model",
  "task_type": "embed",
  "backend": "transformers",
  "quantization_mode": "int8",
  "kv_compression_mode": "standard",
  "kv_required": false,
  "max_context": 8192,
  "streaming_supported": false,
  "chat_template_required": false,
  "hardware_constraints": {"min_gpu_memory_gb": 8}
}
```

## Rerank Models

Register rerankers as `task_type=rerank`. Reranking does not allocate generation
KV and should keep `kv_required=false`.

## Fine-Tuning

Recommended flow:

1. Start from a pretrained base or instruct model.
2. Fine-tune with LoRA for quality-sensitive adapters.
3. Use QLoRA when GPU memory is constrained.
4. Export adapter and base revision together with tokenizer files.
5. Run evaluation before quantization.
6. Quantize to INT4/INT8/FP8 according to target hardware.
7. Register the promoted model version in `config/model_registry.json`.

## Promotion Gate

Promote a model only after:

- structured output conformance passes;
- tool-call JSON format passes;
- latency target passes at expected concurrency;
- KV compression quality delta is within tolerance;
- prefix reuse and memory metrics are visible in Prometheus;
- tenant isolation tests pass.
