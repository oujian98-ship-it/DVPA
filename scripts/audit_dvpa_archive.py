"""Audit an existing trusted local DVPA run with its frozen implementation."""

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    folder = parser.parse_args().archive.resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError("Audit only trusted snapshots stored inside this project")
    snapshots = dict(source_snapshot=folder / "source_snapshot", upstream_source=folder / "upstream_source")
    record = json.loads((folder / "implementation.json").read_text(encoding="utf-8"))
    for key, path in snapshots.items():
        manifest = record["source_sha256"][key]
        files = {p.relative_to(path).as_posix(): p for p in path.rglob("*.py")}
        if set(manifest) != set(files) or any(hashlib.sha256(p.read_bytes()).hexdigest() != manifest[name]
                                              for name, p in files.items()):
            raise ValueError(f"Frozen source manifest mismatch: {key}")
    sources = snapshots["source_snapshot"]
    spec = importlib.util.spec_from_file_location("src", sources / "__init__.py",
                                                   submodule_search_locations=[str(sources)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["src"] = package
    spec.loader.exec_module(package)
    upstream = importlib.import_module("src.upstream_runner")
    upstream.SOURCE = snapshots["upstream_source"]
    for name in ("dvpa_config", "dvpa_public_proxy", "dvpa_public64", "dvpa_train_public", "dvpa_dataset_support", "dvpa_forecast"):
        if (sources / f"{name}.py").is_file():
            module = importlib.import_module(f"src.{name}")
            if hasattr(module, "ROOT"):
                module.ROOT = ROOT
    audit = importlib.import_module("src.dvpa_audit")

    class SourceRoot:
        def __truediv__(self, name):
            return sources if name == "src" else ROOT / name

    audit.ROOT = SourceRoot()
    print(json.dumps(audit.audit(folder), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
