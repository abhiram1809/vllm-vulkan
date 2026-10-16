# LFM2 quantization expansion plan

Completed. The implementation and per-file review map are in `docs/lfm2.md`.
All 50 LFM2 tests pass on RX 560X and Vega 8. Full 2.6B MLX4, Q8_0, Q4_K and
MLX6 qualification passed 268 independent full-vocabulary comparisons; the
reports and hashes are in `docs/validation/lfm2-rx560x*.json`.

Scope is the active weight formats in `flags::QuantFormat`, plus the MLX
2/6/8-bit kernels covered by the existing device conformance suite. Shader
source files alone do not establish a supported checkpoint format.

1. Introduce a typed matrix codec shared by projection and embedding dispatch.
   Cover F32/F16/BF16, MLX affine 2/4/6/8 (group 32/64/128), GGML Q4_0/Q4_K/Q8_0,
   FP8 E4M3 (tensor/row scale), and NVFP4 (E2M1 plus block/global scales).
   Preserve the existing MLX4 fast path. Use FP32/uint32 portable decoding for
   formats whose older kernels require unavailable narrow arithmetic/storage.
2. Load native and mixed-format safetensors, including indexed shards with
   auxiliary tensors in separate files. Validate metadata, dimensions, codec
   alignment, finite coefficients and required pipelines. Split large matrices
   into row chunks that respect Vulkan storage-buffer limits.
3. Add an explicit per-model `quant` residency option for floating formats and
   Q4_0/Q4_K/Q8_0, reusing the repo's canonical quantizers. Keep native packed
   formats in `auto`; never reinterpret one format as another or silently
   change an unsupported request into F16. Provide reproducible conversion
   evidence for the quantized weights used by the reference.
4. Extend the independent Transformers oracle and deterministic fixtures for
   every codec, mixed matrices, group sizes, cross-word MLX6 codes, row chunks,
   shard boundaries and cache/error lifecycle. Test every format on RX 560X
   and Vega 8, including FP16 storage without FP16 arithmetic.
5. Re-run full-model qualification for the baseline INT4 checkpoint and other
   practical packed formats; run existing Rust/Python/conformance checks;
   publish a format/validation matrix with source and checkpoint provenance.

The repo's Q4/Q8 runtime formats use GGML block layouts. Its existing model
loaders read safetensors; adding a general GGUF-container loader is separate
from supporting those runtime codecs. Full F16/BF16/F32 2.6B residency exceeds
this GPU's 4 GiB; format correctness can still be tested on smaller models.
