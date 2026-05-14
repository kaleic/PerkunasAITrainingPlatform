import torch
from training.src.perkunas_training.model.configuration import PerkunasConfig
from training.src.perkunas_training.model.modeling_perkunas import PerkunasForCausalLM
import yaml

# Load config YAML
with open("training/configs/model_3050_pilot.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f)

cfg = PerkunasConfig(**cfg_dict)

# Build model
model = PerkunasForCausalLM(cfg)

# Count params
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total:,}")
print(f"Trainable params: {trainable:,}")
print(f"Approx size fp32: {total * 4 / 1024**2:.2f} MB")
print(f"Approx size fp16: {total * 2 / 1024**2:.2f} MB")
