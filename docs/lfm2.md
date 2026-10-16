# LFM2.5 native Vulkan forward

`vllm_vulkan._rs.Lfm2Model` executes a complete single-sequence LFM2 forward
pass on Vulkan: packed embedding, operator RMSNorm, double-gated short
convolution or Q/K-normalized GQA with split-half RoPE, residual connections,
SwiGLU, final RMSNorm and the tied packed vocabulary projection. Activations,
KV and convolution state remain on the GPU. Full FP32 logits are returned.
Prompt chunks run sequentially and project vocabulary logits only at their end.

The selected checkpoint is
[`mlx-community/LFM2.5-2.6B-4bit`](https://huggingface.co/mlx-community/LFM2.5-2.6B-4bit),
revision `72a08228cf703cd7dc0187abf4edbbd8e935f864`.
Its `model.safetensors` SHA256 is
`c60882ad5ae00906e17b6fe14d1a86d6f63d4e2e2782ea0dced24bb97bef9766`.
This is the 30-layer, 2048-hidden model: 22 convolution blocks, eight attention
blocks, 32 query heads, eight KV heads, 64 dimensions per head, and 128000 tokens.

Matrices retain their native storage format under `quant="auto"`. Small dense
weights and quantization coefficients are converted to FP32 at load time.
Arithmetic uses decoded FP32 weights and FP32 activations; it does not emulate
intermediate BF16 rounding in MLX. The tied embedding and vocabulary projection
share the same allocations. Checkpoint shards are memory-mapped during loading
and the mappings are then released.

## Weight formats

This covers the active `flags::QuantFormat` variants and the MLX widths tested
by the repo's device conformance suite. Both embedding and projection use the
same format decoder; individual matrices may use different formats.

| Native safetensors format | Storage and required coefficients | Explicit `quant` residency |
| --- | --- | --- |
| F32, F16, BF16 | Dense `[rows, columns]` in the named dtype | `f32`, `f16`, `bf16` |
| MLX affine 2/4/6/8-bit | U32 packed rows; `.scales` and `.biases`; affine bits/group metadata, group 32/64/128 | `mlx2`, `mlx4`, `mlx6`, `mlx8` |
| GGML Q4_0, Q4_K, Q8_0 | U8 rows of canonical 18/144/34-byte blocks; explicit per-matrix `format` | `q4_0`, `q4_k`, `q8_0` |
| FP8 E4M3 | F8_E4M3 or explicitly tagged U8; `.weight_scale` scalar or output-row scale | Native `auto` |
| NVFP4 E2M1 | U8 packed nibbles, `.weight_scale` per 16 values, `.weight_scale_2` global multiplier | Native `auto` |

MLX columns must align to their group size. GGML Q4_0/Q8_0 require multiples
of 32 columns; Q4_K requires multiples of 256. FP8 scales may be scalar, `[1]`,
`[1,1]`, `[rows]` or `[rows,1]`. NVFP4 block scales may be FP8 E4M3 or dense
F32/F16/BF16 and must be nonnegative. FP8 E4M3 NaN codes decode to zero, matching
the existing repo decoder. All other non-finite coefficients, malformed shapes,
missing auxiliaries and unsupported format requests fail during loading.

For non-MLX native formats, the explicit metadata form is
`quantization_config.tensor_formats["model.layers.0.feed_forward.w1"] =
{"format": "q4_k"}` (the `language_model.model.` prefix also works). MLX
metadata uses `{"mode":"affine","bits":6,"group_size":128}` and may be
global or per matrix. Indexed safetensors shards may separate weights from
their coefficients. The loader checks the index, duplicates and tensor shapes.
Global exporter metadata may leave embeddings dense in a quantized model;
per-matrix format tags must agree with each tensor's actual storage.

`quant` converts matrices to the requested residency format in bounded row
chunks using the repo's canonical GGML encoders or affine MLX encoding. This
changes quantized values. `weight_formats()` reports the actual resident
formats; an unsupported request raises an error. Export exact packed residency
bytes with `scripts/convert_lfm2.py` for reuse and independent validation.
FP8/NVFP4 are native input formats; this API does not synthesize those formats.
Q4_K uses the repo's CPU reference encoder with block-scale searches, so full
model conversion can be slow on this machine. Export once and reload with
`auto` to avoid repeating that work at every construction.
The observed 2.6B Q4_K CPU export took about 17 minutes during this desktop run.

Large weight matrices are split by rows to fit `maxStorageBufferRange` for
both weights and coefficients. `max_buffer_bytes` optionally lowers that
per-buffer budget for testing. This does not reduce total GPU residency.
Full F32/F16/BF16 2.6B weights exceed the RX 560X's 4 GiB; their forward paths
are qualified on smaller fixtures. GGML codecs here describe runtime/safetensors
storage; this does not introduce a GGUF-container loader or GPTQ/AWQ support.

The new kernels use FP32, uint32 and workgroup shared-memory reductions. They
do not require FP16 arithmetic, narrow storage, integer dot products, subgroup
extended types or wave64. The existing format-specific MLX INT4 matvec is
reused for unchunked MLX4 matrices. Each forward shader module and the selected INT4 variant
must be available before inference. The shared cache also receives two legacy
F16 sources for its eager variant setup; capability filtering can omit those,
and LFM2 never dispatches them. The earlier portability fixes from upstream
commit `3342726`, including descriptor sharing and transfer/host dependencies,
are the base of this change.

## Build and run

Use a Python environment with NumPy, safetensors, Transformers with LFM2
support, and huggingface-hub. Reference tests additionally use PyTorch and
pytest. Building requires Cargo, Vulkan and glslangValidator.

```sh
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/LFM2.5-2.6B-4bit',
    revision='72a08228cf703cd7dc0187abf4edbbd8e935f864',
    local_dir='../models/LFM2.5-2.6B-4bit')
PY
PYO3_PYTHON=../.venv312/bin/python bash scripts/build_lfm2.sh
PYTHONPATH=. ../.venv312/bin/python scripts/run_lfm2.py \
  ../models/LFM2.5-2.6B-4bit --device 1 --max-new-tokens 128
```

Device 1 is the RX 560X on the development machine; enumerate devices with
`vllm_vulkan._rs.enumerate_devices()` before choosing on another machine.

```python
from vllm_vulkan._rs import Lfm2Model

model = Lfm2Model("/path/to/checkpoint", device_idx=1, max_seq_len=512)
logits = model.forward_prefill([1, 9, 17])  # appended positions 0, 1, 2
logits = model.forward(31, position=3)  # full logits for the next token
model.reset_kv_cache()  # next append begins at zero

# Explicit residency conversion; auto preserves each matrix's native format.
del model
model = Lfm2Model("/path/to/checkpoint", 1, 512, quant="q8_0")
print(model.weight_formats())
```

Export and run a reusable converted checkpoint:

```sh
PYTHONPATH=. ../.venv312/bin/python scripts/convert_lfm2.py \
  ../models/LFM2.5-2.6B-4bit ../models/LFM2.5-derived-q8 --quant q8_0
PYTHONPATH=. ../.venv312/bin/python scripts/run_lfm2.py \
  ../models/LFM2.5-derived-q8 --device 1
```

Conversion requires a new destination directory. Converting an existing INT4
checkpoint into Q8 preserves information from that INT4 source; it cannot
recover original dense-model quality. Reports identify both source and reference
weights so codec correctness and quantization quality remain distinguishable.

Each instance owns one sequence and must remain on its creating thread.
`position` must equal the current cache length. Empty chunks, invalid token IDs,
nonconsecutive positions and capacity overflow fail before any cache mutation.
Reset makes stale KV unreachable and the first convolution step clears old
history. GPU recording/submission errors invalidate the instance; recreate it
instead of reusing potentially partial state.

The loader supports local single-file or indexed sharded safetensors with MLX
`language_model.model.` or HF `model.` names, the formats above, tied embeddings,
bias-free convolutions and default unscaled RoPE. It validates tensor shapes,
storage dtypes and finite coefficients. Other architectures/configurations are
rejected. The chosen cache capacity
must fit the device storage-buffer and memory limits. This standalone engine
has a separate API from vLLM's scheduler; scheduler integration, sequence
batching and a tiled parallel prefill are outside this forward-pass change.

## Validation

```sh
PYO3_PYTHON=../.venv312/bin/python cargo test \
  --no-default-features --features multiple-pymethods,lfm2 --lib \
  -- --skip device_conformance
VLLM_VULKAN_TEST_DEVICE=1 PYTHONPATH=. ../.venv312/bin/python -m pytest \
  tests/python/test_lfm2.py -v
PYTHONPATH=. ../.venv312/bin/python scripts/validate_lfm2.py \
  --model ../models/LFM2.5-2.6B-4bit --device 1 --length 64 \
  --output ../results/lfm2-rx560x.json
```

For the bundled Python runtime on this machine, Rust tests also need its
actual library directory (the bundled build's sysconfig reports `/install/lib`):

```sh
LFM_PY_LIB=$(../.venv312/bin/python -c 'import sys; print(sys.base_prefix+"/lib")')
export LIBRARY_PATH="$PWD/../.native-libs" LD_LIBRARY_PATH="$LFM_PY_LIB"
```

The oracle runs the installed Transformers `Lfm2ForCausalLM` with eager
attention and explicit FP32 dequantization, streaming weight chunks to keep
host memory bounded. It does not call the Rust forward or use its kernels.
Tests compare every vocabulary logit and exact greedy IDs, with relative L2
below `5e-5`. The full-checkpoint qualification additionally requires cosine
above `0.999999`. Tiny fixtures cover 32/64/128/256-dimensional heads,
convolution lengths 1/3/4, all supported coefficient dtypes, sequential and
chunked calls, cache reset, rejection atomicity and capacity boundaries. The
format suite additionally covers every matrix codec, all MLX group sizes,
MLX6 cross-word packing, mixed formats, separated auxiliary shards, constrained
row buffers, explicit conversion and reloading the exported packed bytes. A
deliberately perturbed answer must fail the comparator.

For a residency conversion, export the reference first, then compare GPU
conversion against an independent decoder of those exported bytes:

```sh
PYTHONPATH=. ../.venv312/bin/python scripts/validate_lfm2.py \
  --model ../models/LFM2.5-2.6B-4bit --device 1 --length 64 --quant q8_0 \
  --reference-model ../models/LFM2.5-derived-q8 \
  --output ../results/lfm2-rx560x-q8.json
```

## Review map for the format expansion

| File | Change |
| --- | --- |
| `src/lfm2_quant.rs` | Typed codecs, strict native loader, indexed shards, canonical residency conversion/export, storage-limit row splitting and projection/embedding dispatch |
| `src/lfm2.rs` | Uses typed matrices throughout the existing graph; adds `quant`, `group_size`, `max_buffer_bytes`, `weight_formats()` and `convert_checkpoint()` |
| `shaders/lfm2_weights.glsl` | One portable scalar decoder for all twelve format families |
| `shaders/lfm2_matvec.comp`, `shaders/lfm2_embed.comp` | Shared decoding for all projections and embeddings, including chunk row offsets |
| `scripts/lfm2_formats.py`, `scripts/lfm2_reference.py` | Independent decoders, mixed/sharded fixtures and streamed Transformers weights |
| `scripts/convert_lfm2.py`, `scripts/run_lfm2.py`, `scripts/validate_lfm2.py` | Conversion and residency CLI options; source/reference shard hashes and actual resident formats in qualification reports |
| `tests/python/test_lfm2.py` | All codecs/groups, HF/MLX names, scale guards, shard separation, row limits, conversion/reload and lifecycle checks |
| `src/lib.rs`, `src/pipeline.rs`, `scripts/compile_shaders.sh` | Register the module/kernel and verify portable shader capabilities |

The implementation sequence and scope are recorded in
[the quantization plan](lfm2-quant-plan.md).

## Local verification results

| Check | Result |
| --- | --- |
| LFM2 feature Rust suite | 153 passed, 13 existing opt-in tests ignored |
| Default-model Rust suite | 469 passed, 46 existing opt-in tests ignored |
| Python suite, excluding slow tests | 170 passed, 3 dependency skips, 5 slow tests deselected |
| LFM2 release tests on RX 560X | All 50 tests passed, included in the Python suite |
| LFM2 release tests on Vega 8 | All 50 tests passed |
| Existing Vulkan conformance on RX 560X | 12 passed |
| Existing Vulkan conformance on Vega 8 | 12 passed |
| Full 2.6B native MLX4 on RX 560X | 103 full-vocabulary comparisons; worst relative L2 `1.20e-5`; exact greedy IDs |
| Full 2.6B native Q8_0 on RX 560X | 55 comparisons; worst relative L2 `3.32e-6`; exact greedy IDs |
| Full 2.6B native Q4_K on RX 560X | 55 comparisons; worst relative L2 `3.53e-6`; exact greedy IDs |
| Full 2.6B native MLX6 on RX 560X | 55 comparisons; worst relative L2 `1.05e-5`; exact greedy IDs |
| New Python lint/format, Rust source formatting, shell syntax, diff whitespace | Passed |
| LFM2-feature Clippy | Completed without errors; advisory and existing warnings remain |

The full-checkpoint reference uses PyTorch `2.14.1+cpu` and Transformers
`5.19.0`. The reference's PyTorch convolution fallback is intentional: no
accelerated convolution package is required to independently validate the GPU.

[The saved qualification report](validation/lfm2-rx560x.json) records the
checkpoint, binary and runtime-source hashes, input IDs, device and per-check
errors. Across 13,184,000 compared logits, the worst relative L2 error is
`1.19756e-5`, below the `5e-5` bound. Checks include a 64-token raw prompt,
chunked prefill, cache reset, continuation after rejected requests, and 32
greedy chat decode steps checked against a second independent teacher-forced
Transformers run.

Additional reports cover the canonical converted
[Q8_0](validation/lfm2-rx560x-q8_0.json),
[Q4_K](validation/lfm2-rx560x-q4_k.json) and
[MLX6](validation/lfm2-rx560x-mlx6.json) checkpoints. These use the same pinned
INT4 source and independently decoded exported weights. Each checks a 16-token
raw sequence, chunk/reset/error recovery and 32 chat decode steps. All 167
logical matrices per model use the named resident format. The four final
reports cover 34,304,000 logits across 268 comparisons and share the final
release binary hash. Every codec, including the formats without a full 2.6B
run, is covered by the smaller Transformers forward tests on both GPUs.

The fixed 32-step decode interval observed 11.123 tokens/s, including Python
argmax and full-logit readback. This functional qualification is a short desktop
run, not a throughput boundary or a comparison with another implementation.
