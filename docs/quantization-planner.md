# Quantization Planner Design

The quantization planner lives in `src/kvserve/quantization/planner.py`.

## Inputs

- model registry entry;
- policy mode;
- detected hardware profile;
- pre-quantized model metadata.

## Hardware Detection

The planner detects:

- CUDA vs ROCm vs CPU;
- GPU name;
- total GPU memory;
- compute capability;
- BF16 support;
- FP8 support;
- INT4 support;
- device count.

FP8 is selected only when hardware advertises suitable CUDA capability. INT4 is
selected only for GPU-capable backends or explicit pre-quantized formats.

## Modes

Supported weight modes:

- BF16
- FP16
- FP8
- INT8
- INT4 AWQ-style
- AUTO

Supported policy modes:

- `quality_first`: BF16/FP16 before low-bit modes.
- `balanced`: FP8 when supported, INT4 when memory constrained, otherwise BF16/FP16.
- `memory_first`: INT4 first, INT8 fallback.
- `throughput_first`: FP8 first, INT4 fallback.

## Output

`QuantizationPlan` includes:

- selected mode;
- weight dtype;
- load format;
- KV cache dtype;
- whether online quantization is required;
- whether the model is pre-quantized;
- reason string for audit and debugging.

The Transformers adapter turns INT8/INT4 plans into `BitsAndBytesConfig`. The
vLLM adapter maps selected quantization and FP8 KV into engine arguments.
