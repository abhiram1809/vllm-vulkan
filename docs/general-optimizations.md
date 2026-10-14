# Hardware-independent runtime optimizations

This branch extracts changes that can apply to the existing runtime without
selecting a GPU vendor, device index, wave size, tile geometry or model-specific
execution graph. It starts from fork `main` at `fbccc82a49bf5e3a08ea41e62b5e92acb25f6ebd`.

## Included changes

- **Fused CPU embedding conversion.** `PackedEmbed::row_f32` writes its FP32
  result directly, removing the intermediate FP16 vector. Four-bit group64
  rows use constant shifts and masks automatically. Other formats keep their
  original unpacking math. Both paths retain separate FP32 affine operations
  and the original FP16 rounding before conversion back to FP32.
- **Allocation-free shared dispatch recording.** `record_to` and `record_to_off`
  share descriptor setup and command recording through an iterator-based helper.
  Both use bounded stack arrays; offset dispatches no longer allocate two vectors.
  Both reject excess bindings before consuming a descriptor set.
- **Device capabilities drive shader availability.** Query and enable supported
  integer arithmetic, integer-dot and narrow-storage features. Inspect SPIR-V
  capabilities and subgroup operand/result types before startup, eager and
  timeout-isolated variant compilation. Extended-type subgroup operations
  require [`shaderSubgroupExtendedTypes`](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#VUID-RuntimeSpirv-None-06275); FP16 storage and FP32 subgroup arithmetic
  alone do not. Unsupported optional shaders remain absent;
  a genuinely missing required source still produces an error.
- **Availability-aware GEMM dispatch.** The shared and Qwen GEMM callers,
  including the fused Qwen FFN, use a preferred geometry only when its pipeline
  survived feature filtering. Otherwise they select `matmul_f16_f32_fp32` with
  its matching 64×64 tiles. Devices with FP16 storage but no FP16 arithmetic
  can therefore execute these projections after successful initialization.
- **Weight-format-safe GEMM inputs.** Tiled F16 GEMMs accept only plain F16
  weights without auxiliary buffers, including all three fused FFN projections.
  Other formats use the active architecture's host weights. If lean-host mode
  omitted those tensors, the existing per-format GPU matvecs serve each token
  instead. Qwen3.5 F32 residency likewise uses its format-aware projection path.
- **Aligned SPIR-V loading.** Eager and timeout-isolated compilation copy byte
  slices into aligned words with `ash::util::read_spv`; they no longer construct
  potentially unaligned `&[u32]` slices.
- **Explicit transfer and host dependencies.** Order uploads before compute,
  compute before downloads, and device writes before mapped host reads. Compute
  and transfer dependencies include shader writes for in-place consumers. Batch
  host dependencies cover both shader writes and transfer-written snapshots
  across synchronous, asynchronous and ring submissions. Host
  coherency does not replace a memory dependency; see the
  [Khronos synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html).
- **Selectable conformance device.** `VLLM_VULKAN_TEST_DEVICE` defaults to zero
  and lets the same suite test another adapter on a multi-GPU machine.

These changes do not introduce new Python APIs or inference flags. The compiled
shader sources and each pipeline's existing tile geometry are unchanged.

## Evaluation of the remaining experiment changes

| Experiment change | Assessment for this branch |
| --- | --- |
| Polaris tiles, register choices, wave64 kernels, larger-batch selection | Device-dependent; excluded. |
| RX560X VRAM percentages, explicit VRAM mirrors, clock telemetry | Machine or memory-policy experiments; excluded. |
| Native Qwen forward/prefill, fixed-slot batching, shared KV, partial greedy head | Useful feature work, but dependent on new execution contracts and subgroup-sensitive attention; needs a separate inference PR. |
| FP32 recurrent history | The algebra is portable, but the implemented capacity/state contract depends on the new native batch runtime; deferred with that feature. |
| Lossless BF16 auxiliary storage, GEMM prefetch and split-K variants | Potentially portable kernels; the current loader/selector integration and performance evidence are specialized. They need separate shape/device qualification rather than inheriting RX560X defaults. |
| GPU embedding lookup | Depends on the new tied-weight native batch graph and explicit floating-point controls; deferred with that feature. |
| Canonical GDN epsilon and prefix invalidation | Hardware-independent correctness change, but changes recurrent semantics and saved-prefix compatibility; belongs in a separate correctness PR. |
| Benchmark-only shader variants and exploratory APIs | Excluded from the runtime change. |

## Verification

Run the existing feature slices with the new regression tests:

```sh
cargo test --no-default-features --features multiple-pymethods,qwen35 --lib -- --skip device_conformance
cargo test --no-default-features --features multiple-pymethods --lib -- --skip device_conformance
cargo test --no-default-features --features multiple-pymethods,gemma,qwen35 --lib -- --skip device_conformance
cargo check --tests
VLLM_VULKAN_TEST_DEVICE=0 cargo test --no-default-features --features multiple-pymethods,gemma,qwen35 --lib device_conformance -- --nocapture
```

The optional `debug-api` feature references `src/debug_api.rs`, which is absent
in the fork base. Consequently `--all-features` is not a supported build on
that base; the default model features and the foundation/Qwen slices are the
verified configurations for this branch.

CPU tests cover rounding boundaries, signed zeros, subnormals, overflow,
non-finite coefficients and multiple bit widths/group sizes. Capability tests
cover missing features, storage-only FP16, malformed modules and unaligned
byte slices. GPU tests compare the existing quantized kernels against CPU
references and exercise both zero-offset entry points, aligned offset descriptors,
in-place updates, untouched neighboring slices and a valid dispatch after rejecting
excess bindings through either entry point.
GEMM selection tests cover missing preferred pipelines and preservation of
available geometries. GPU tests disable FP16 arithmetic in the cache's
capabilities and execute the selected FP32 fallback across partial tiles,
checking every output against a CPU reference.
Caller-level tests reject non-F16 and auxiliary-buffer layouts, check every
fused FFN weight, preserve the plain-F16 GPU route, and compare lean-host
Q4/Q8/BF16/F32 projections against decoded CPU references.

A manually enabled CPU checkpoint audit and paired benchmark is also available:

```sh
VLLM_VULKAN_EMBED_BENCH_MODEL=/path/to/model.safetensors \
  cargo test --release --no-default-features --features multiple-pymethods,qwen35 \
  --lib packed_embed_checkpoint_benchmark -- --ignored --nocapture
```

It requires a local group64 MLX4 checkpoint with BF16 affine coefficients. It
checks every embedding value against the original `row_f16` round-trip before
warming both paths and measuring eight alternating runs. No download or GPU
initialization occurs. Numerical equivalence is required; speedup is reported
for the measured CPU rather than asserted for every architecture.

## Local results

| Check | Result |
| --- | --- |
| Qwen feature CPU suite | 230 passed; GPU tests run separately |
| Foundation CPU suite | 148 passed; GPU tests run separately |
| Combined Gemma/Qwen CPU suite | 300 passed; GPU tests run separately |
| Default production model features and tests | `cargo check --tests` passed |
| RX560X conformance (Gemma/Qwen features) | 12 tests passed; 18 quant kernels, GEMM fallback and weight-format callers exercised; FP16 subgroup isolation skipped |
| Vega 8 conformance (Gemma/Qwen features) | 12 tests passed; 20 quant kernels, GEMM fallback and weight-format callers exercised; FP16 subgroup isolation exercised |
| Checkpoint embedding audit | All 508,559,360 values bit-identical |
| Paired CPU embedding benchmark | 2.553x measured median speedup |

The checkpoint audit uses `mlx-community/Qwen3.5-2B-MLX-4bit`, revision
`93760be4f1f69842a46bc13dbdc0f19e291392a3`, with safetensors SHA256
`713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5`.
For 10,000 decoded rows, four baseline observations are 0.218036, 0.218165,
0.219188 and 0.227584 seconds; four fused observations are 0.080726,
0.080883, 0.090399 and 0.090654 seconds. The benchmark warms both paths and
alternates them in ABBA order. These are CPU embedding timings, not full-model
or cross-device throughput claims.
