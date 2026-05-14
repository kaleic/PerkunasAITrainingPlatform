import training.src.perkunas_training.model.configuration as cfgmod
import training.src.perkunas_training.model.modeling_perkunas as modelmod

print("configuration.py exports:")
print([n for n in dir(cfgmod) if not n.startswith("_")])

print("\nmodeling_perkunas.py exports:")
print([n for n in dir(modelmod) if not n.startswith("_")])
