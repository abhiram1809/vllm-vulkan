#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYO3_PYTHON="${PYO3_PYTHON:-python3}"
cargo build --release --no-default-features --features multiple-pymethods,lfm2,pyo3/extension-module
staged_extension=$(mktemp vllm_vulkan/_rs.so.XXXXXX)
trap 'rm -f -- "$staged_extension"' EXIT
cp -p target/release/lib_rs.so "$staged_extension"
"$PYO3_PYTHON" - "$staged_extension" <<'PY'
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
loader = importlib.machinery.ExtensionFileLoader("vllm_vulkan._rs", str(path))
spec = importlib.util.spec_from_file_location(loader.name, path, loader=loader)
if spec is None:
    raise ImportError(f"Cannot load staged extension: {path}")
module = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = module
loader.exec_module(module)
from vllm_vulkan import _rs

print(_rs.enumerate_devices())
PY
mv "$staged_extension" vllm_vulkan/_rs.so
