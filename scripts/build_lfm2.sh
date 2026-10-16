#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYO3_PYTHON="${PYO3_PYTHON:-python3}"
cargo build --release --no-default-features --features multiple-pymethods,lfm2,pyo3/extension-module
cp target/release/lib_rs.so vllm_vulkan/_rs.so.new
mv vllm_vulkan/_rs.so.new vllm_vulkan/_rs.so
"$PYO3_PYTHON" -c 'from vllm_vulkan import _rs; print(_rs.enumerate_devices())'
