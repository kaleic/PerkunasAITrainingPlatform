from __future__ import annotations

import pytest
import torch

from perkunas_training.model.configuration import PerkunasConfig
from perkunas_training.model.modeling_perkunas import PerkunasForCausalLM
from perkunas_training.train.device import assert_model_device, select_device, verify_batch_devices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available in this runtime")
def test_model_forward_is_on_cuda() -> None:
    device = select_device(require_gpu=True)
    config = PerkunasConfig(
        model_name="gpu-test",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=16,
    )
    model = PerkunasForCausalLM(config).to(device)
    model_device = assert_model_device(model, require_gpu=True)
    assert model_device.type == "cuda"

    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
    labels = torch.randint(0, config.vocab_size, (2, 16), device=device)
    verify_batch_devices(
        input_ids=input_ids,
        labels=labels,
        expected_device=device,
        require_gpu=True,
    )
    output = model(input_ids=input_ids, labels=labels)
    assert output["loss"].device.type == "cuda"
