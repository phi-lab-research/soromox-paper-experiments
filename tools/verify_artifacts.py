"""Check that the archived paper data and outputs match the migration baseline."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    artifacts = json.loads((root / "provenance/migration.json").read_text())["artifacts"]
    failures = []
    for name, expected in artifacts.items():
        path = root / name
        if not path.is_file():
            failures.append(f"missing: {name}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
            failures.append(f"changed or unresolved LFS pointer: {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(artifacts)} paper artifacts against the migration baseline.")


if __name__ == "__main__":
    main()
