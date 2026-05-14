from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvserve.models.schemas import ModelSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="config/model_registry.json")
    parser.add_argument("--model-json", required=True, help="Path to a JSON object matching ModelSpec")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    model_path = Path(args.model_json)
    document = json.loads(registry_path.read_text(encoding="utf-8"))
    model = ModelSpec.model_validate_json(model_path.read_text(encoding="utf-8"))
    models = [entry for entry in document.get("models", []) if entry.get("model_id") != model.model_id]
    models.append(model.model_dump(mode="json"))
    document["models"] = models
    registry_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"registered {model.model_id} in {registry_path}")


if __name__ == "__main__":
    main()
