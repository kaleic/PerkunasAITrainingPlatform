from __future__ import annotations

import json
from pathlib import Path

from kvserve.app import create_app


def main() -> None:
    schema = create_app().openapi()
    out = Path("docs/openapi.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
