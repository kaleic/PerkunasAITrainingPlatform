# Perkunas Streaming Training: A 100M Parameter Language Model Under an 8GB VRAM Limit

## Abstract

Perkunas is a streaming-first training architecture designed to push language model training beyond the practical memory limits of constrained hardware.

In this public experiment, a 100M parameter language model was trained on the TinyStories dataset using an NVIDIA GeForce RTX 3050 with 8GB of VRAM. The run included full training updates, validation during training, optimizer state, checkpoint recovery, and continued improvement after a long plateau.

The model reached a TinyStories validation loss of `3.5135`, with validation perplexity falling to `33.57` at the best published checkpoint in this run.

This is not presented as a state-of-the-art TinyStories benchmark. It is presented as a systems milestone: a 100M parameter model trained end-to-end under a memory limit that would normally be considered heavily constrained for this class of work.

The early result is simple:

> The architecture works. The model learns. The hardware did not stop it.

## Key Result

The model crossed below `4.0` validation loss and continued improving into the mid-`3.x` range while training under an 8GB VRAM limit.

| Step | Validation Loss | Validation Perplexity |
|---:|---:|---:|
| 100 | 6.8512 | 944.99 |
| 200 | 5.6404 | 281.59 |
| 300 | 5.1590 | 174.00 |
| 400 | 4.9266 | 137.91 |
| 500 | 4.8817 | 131.85 |
| 600 | 4.7575 | 116.46 |
| 800 | 4.7376 | 114.16 |
| 1500 | 4.7692 | 117.83 |
| 1600 | 4.1140 | 61.19 |
| 1700 | 3.9910 | 54.11 |
| 1800 | 3.9547 | 52.18 |
| 1900 | 3.9178 | 50.29 |
| 2000 | 3.9007 | 49.44 |
| 2100 | 3.8389 | 46.48 |
| 2200 | 3.7610 | 42.99 |
| 2400 | 3.5857 | 36.08 |
| 2800 | 3.5414 | 34.52 |
| 2900 | 3.5214 | 33.83 |
| 3000 | 3.5135 | 33.57 |

The most important point is not only the final number. It is the shape of the run.

The model spent hundreds of steps near a difficult plateau in the high `4.x` range, then broke through into a new regime:

- Step 1500: `4.7692`
- Step 1600: `4.1140`
- Step 1700: `3.9910`
- Step 3000: `3.5135`

That late-stage movement matters. It shows the system did more than survive the first easy part of training. It continued improving after optimization became harder.

## Why This Matters

Language model training is usually gated by memory. Raw model weights are only part of the cost. Training also requires gradients, optimizer state, activations, validation, checkpointing, and recovery.

On small GPUs, those requirements quickly become the wall.

Perkunas takes a different path. It is built around a streaming training model that allows constrained hardware to participate in larger experiments than it would normally support.

The public result:

> A 100M parameter language model trained and validated on an RTX 3050 with 8GB of VRAM.

That is the center of this release.

This experiment suggests that the hardware barrier for model development can be pushed downward. Memory-constrained systems across deployment classes can become more useful when the training runtime is designed around memory limits from the beginning.

## Machine Limits

The experiment was intentionally run under a tight VRAM limit:

- GPU: NVIDIA GeForce RTX 3050
- VRAM: 8GB
- Dataset: TinyStories
- Sequence length: 512 tokens
- Model scale: 100M parameters
- Training mode: Perkunas streaming runtime

This hardware profile is part of the result. The point was not to show that an unconstrained accelerator can train a language model. The point was to show that a memory-limited system can train a meaningful model when paired with a training architecture built for memory pressure.

## Architecture Effectiveness

The internal mechanics of Perkunas are not described in this public note. The implementation is proprietary and still evolving.

What can be shown publicly is the behavior:

- The model trained end-to-end.
- Validation ran throughout training.
- Checkpoints were preserved and resumed.
- The run recovered from a plateau.
- Validation loss continued falling.
- The system trained a 100M parameter model under an 8GB VRAM limit.

That behavioral evidence is the core claim.

Perkunas is not only a model. It is a training architecture. The current run is an early demonstration that the architecture can convert constrained hardware into a real training environment.

## Training and Serving

This release separates two goals that are often blended together:

- training under severe memory pressure;
- serving a frozen checkpoint with low latency.

Perkunas is designed to solve the first problem. After training, the checkpoint can be packaged into a standard inference artifact for mature serving runtimes. That means the training system can remain focused on memory-efficient learning, while deployment can use a runtime optimized for fast decoding when the target machine has enough inference memory.

This does not reduce the training result. It makes the design more practical:

> Train in the streaming architecture. Serve from a packaged checkpoint when speed matters.

The current model can be exported into a Hugging Face-compatible package for standard text-generation infrastructure. The public point is not that Perkunas must replace every inference server. The point is that Perkunas can make the training run possible, then hand the frozen model to the best serving stack available.

## Training Behavior

The run showed three phases:

1. **Initial learning:** validation loss dropped rapidly from `6.85` into the low `5.x` range.
2. **Plateau:** the model spent several checkpoints near `4.7` validation loss.
3. **Breakthrough:** the model moved from `4.7692` at step 1500 to `3.9910` at step 1700.

This is the part of the curve that makes the result exciting. The model did not merely produce early progress and stall forever. Continued training found a better regime.

## What Was Proven

This experiment demonstrates:

- A 100M parameter language model can train under an 8GB VRAM limit.
- A streaming-first training architecture can support meaningful model learning under memory pressure.
- Validation loss can continue improving after a long plateau.
- Training can be checkpointed, resumed, and continued during active experimentation.
- A trained streaming checkpoint can be packaged for standard inference runtimes.
- Memory-constrained hardware can be made more useful for language model research than its raw memory budget suggests.

## What This Is Not

This release is not claiming:

- state-of-the-art TinyStories performance;
- parity with datacenter GPU training;
- a final production model;
- disclosure of the internal Perkunas implementation;
- that throughput is already optimized.
- that the released checkpoint is instruction-tuned or production-quality for chat.

The goal is to share the public milestone: a real training curve from a 100M parameter model trained under an 8GB VRAM limit, plus a practical path from training checkpoint to standard serving artifact.

## Public Takeaway

Perkunas shows that training capability is not only a question of hardware size. It is also a question of runtime design.

This first public result shows a 100M parameter model training, validating, recovering from a plateau, crossing below `4.0`, and reaching `3.5135` validation loss under an 8GB VRAM limit.

That is the beginning, not the ceiling.

The next steps are to keep pushing the run, publish generation samples, improve throughput, and test larger or more demanding training targets under the same streaming-first approach.

## Next Steps

Planned work includes:

- continuing the TinyStories training run beyond the sub-`4.0` milestone;
- publishing sample generations from multiple checkpoints;
- reporting memory and throughput behavior more formally;
- comparing against conventional memory-resident training limits;
- evaluating larger model scales under the same streaming runtime;
- improving speed while preserving the low-memory training behavior;
- validating exported serving packages with mature inference runtimes;
- preparing a public demo checkpoint once the run matures further.

## Citation

If referencing this experiment, please cite it as:

```text
Perkunas Streaming Training Runtime, TinyStories 100M Parameter 8GB GPU Experiment, 2026.
```
