from __future__ import annotations

import torch

from perkunas_training.model.configuration import PerkunasConfig
from perkunas_training.model.modeling_perkunas import PerkunasForCausalLM
from perkunas_training.train.device import (
    assert_model_device,
    log_first_batch_verification,
    log_gpu_memory,
    select_device,
    verify_batch_devices,
)


def main() -> None:
    device = select_device(require_gpu=True)
    config = PerkunasConfig(
        model_name="perkunas-gpu-smoke",
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    model = PerkunasForCausalLM(config).to(device)
    assert_model_device(model, require_gpu=True)
    log_gpu_memory("GPU memory allocated after smoke model load")

    input_ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 32), device=device)
    verify_batch_devices(
        input_ids=input_ids,
        labels=labels,
        expected_device=device,
        require_gpu=True,
    )
    log_first_batch_verification(
        model=model,
        input_ids=input_ids,
        labels=labels,
        batch_size=2,
        require_gpu=True,
    )
    output = model(input_ids=input_ids, labels=labels)
    output["loss"].backward()
    assert output["loss"].device.type == "cuda"
    log_gpu_memory("GPU memory allocated after forward/backward")
    print("Perkunas CUDA smoke passed", flush=True)


if __name__ == "__main__":
    main()
