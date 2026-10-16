"""Full-logit tests against HF Transformers, including recurrent/KV lifecycle."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.lfm2")
from vllm_vulkan import _rs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from lfm2_reference import reference_model, tiny_checkpoint  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(_rs, "Lfm2Model"), reason="build with lfm2 feature"
)


@pytest.fixture(scope="module")
def device():
    devices = _rs.enumerate_devices()
    if not devices:
        pytest.skip("no Vulkan device")
    torch.set_num_threads(2)
    return int(os.environ.get("VLLM_VULKAN_TEST_DEVICE", "0"))


def compare(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert np.isfinite(actual).all()
    err = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-20)
    assert err < 5e-5, (err, np.max(np.abs(actual - expected)))
    assert actual.argmax() == expected.argmax()


@pytest.mark.parametrize(
    "length,hidden,heads,kvheads,dtype",
    [
        (3, 128, 4, 2, torch.bfloat16),
        (1, 64, 1, 1, torch.float32),
        (4, 256, 2, 1, torch.float16),
        (3, 256, 1, 1, torch.bfloat16),
    ],
)
def test_forward_against_transformers(
    tmp_path, device, length, hidden, heads, kvheads, dtype
):
    checkpoint = tiny_checkpoint(
        tmp_path,
        length=length,
        hidden=hidden,
        heads=heads,
        kvheads=kvheads,
        coeff_dtype=dtype,
    )
    ref = reference_model(checkpoint)
    gpu = _rs.Lfm2Model(str(checkpoint), device, 20)
    ids = [1, 9, 17, 31, 42, 7, 108, 6, 4, 33, 26, 12, 3, 19, 57, 92, 65, 14, 22, 10]
    with torch.inference_mode():
        expected = ref(torch.tensor([ids]), use_cache=False).logits[0].numpy()
    for pos, token in enumerate(ids):
        compare(gpu.forward(token, pos), expected[pos])
    assert gpu.position == 20
    with pytest.raises(RuntimeError, match="max_seq_len"):
        gpu.forward(8)
    assert gpu.position == 20
    gpu.reset_kv_cache()
    compare(gpu.forward_prefill(ids[:7]), expected[6])
    compare(gpu.forward_prefill(ids[7:12]), expected[11])
    # Invalid requests must not alter either convolution or KV state.
    for operation in [
        lambda: gpu.forward(128),
        lambda: gpu.forward(1, 0),
        lambda: gpu.forward_prefill([]),
        lambda: gpu.forward_prefill([3, 128]),
        lambda: gpu.forward_prefill([1] * 9),
    ]:
        with pytest.raises(RuntimeError):
            operation()
        assert gpu.position == 12
    compare(gpu.forward_prefill(ids[12:]), expected[-1])
    gpu.reset_kv_cache()
    new_ids = [5, 4, 3, 2]
    with torch.inference_mode():
        new_expected = (
            ref(torch.tensor([new_ids]), use_cache=False).logits[0, -1].numpy()
        )
    compare(gpu.forward_prefill(new_ids), new_expected)
    # Comparator must reject a deliberately corrupted answer.
    with pytest.raises(AssertionError):
        compare(new_expected + 0.01, new_expected)


def test_checkpoint_guards(tmp_path, device):
    from safetensors.torch import load_file, save_file

    checkpoint = tiny_checkpoint(tmp_path)
    path = checkpoint / "config.json"
    original = json.loads(path.read_text())
    for key, value in [
        ("model_type", "qwen3"),
        ("conv_bias", True),
        ("tie_word_embeddings", False),
        ("quantization", {"bits": 8, "group_size": 64, "mode": "affine"}),
    ]:
        bad = dict(original, **{key: value})
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            _rs.Lfm2Model(str(checkpoint), device, 16)
    path.write_text(json.dumps(original))
    for capacity in [0, 129]:
        with pytest.raises(ValueError, match="max_seq_len"):
            _rs.Lfm2Model(str(checkpoint), device, capacity)
    weights = load_file(checkpoint / "model.safetensors")
    key = "language_model.model.embed_tokens.weight"
    packed = weights[key]
    weights[key] = packed.to(torch.float32)
    save_file(weights, checkpoint / "model.safetensors")
    with pytest.raises(ValueError, match="shape/dtype"):
        _rs.Lfm2Model(str(checkpoint), device, 16)
    weights[key] = packed
    scale_key = "language_model.model.layers.0.conv.in_proj.scales"
    scales = weights.pop(scale_key)
    save_file(weights, checkpoint / "model.safetensors")
    with pytest.raises(ValueError, match="scales"):
        _rs.Lfm2Model(str(checkpoint), device, 16)
    for bad_scales, message in [
        (scales.to(torch.uint32), "F32/F16/BF16"),
        (scales[:, :-1].contiguous(), "shape/dtype"),
        (torch.full_like(scales, float("nan")), "non-finite"),
    ]:
        weights[scale_key] = bad_scales
        save_file(weights, checkpoint / "model.safetensors")
        with pytest.raises(ValueError, match=message):
            _rs.Lfm2Model(str(checkpoint), device, 16)


NATIVE_FORMATS = [
    (f"mlx{bits}", group) for bits in (2, 4, 6, 8) for group in (32, 64, 128)
] + [
    (name, 64)
    for name in (
        "f32",
        "f16",
        "bf16",
        "q4_0",
        "q4_k",
        "q8_0",
        "fp8",
        "fp8_tensor",
        "nvfp4",
    )
]


def qualify_short(gpu, ref):
    ids = [1, 9, 37, 63, 127, 4, 19, 41, 33, 57, 2, 17]
    with torch.inference_mode():
        expected = ref(torch.tensor([ids]), use_cache=False).logits[0].numpy()
    for pos, token in enumerate(ids):
        compare(gpu.forward(token, pos), expected[pos])
    gpu.reset_kv_cache()
    compare(gpu.forward_prefill(ids[:5]), expected[4])
    for invalid in ([128], [], [2, 128], [1] * 20):
        with pytest.raises(RuntimeError):
            gpu.forward_prefill(invalid)
        assert gpu.position == 5
    compare(gpu.forward_prefill(ids[5:]), expected[-1])
    gpu.reset_kv_cache()
    compare(gpu.forward_prefill(ids), expected[-1])


@pytest.mark.parametrize("kind,group", NATIVE_FORMATS)
def test_native_quant_formats(tmp_path, device, kind, group):
    checkpoint = tiny_checkpoint(
        tmp_path,
        hidden=256,
        heads=4,
        kvheads=2,
        intermediate=512,
        quant=kind,
        group_size=group,
    )
    ref = reference_model(checkpoint)
    gpu = _rs.Lfm2Model(str(checkpoint), device, 20)
    label = (
        f"{kind}_g{group}"
        if kind.startswith("mlx")
        else ("fp8" if kind.startswith("fp8") else kind)
    )
    assert set(gpu.weight_formats()) == {label}
    qualify_short(gpu, ref)


@pytest.mark.parametrize("hf_names", [False, True])
def test_mixed_sharded_and_chunked(tmp_path, device, hf_names):
    checkpoint = tiny_checkpoint(
        tmp_path,
        hidden=256,
        heads=4,
        kvheads=2,
        intermediate=512,
        mixed=True,
        sharded=True,
        hf_names=hf_names,
    )
    if hf_names:
        config_file = checkpoint / "config.json"
        config = json.loads(config_file.read_text())
        formats = config["quantization_config"]["tensor_formats"]
        config["quantization_config"]["tensor_formats"] = {
            key.removeprefix("model."): value for key, value in formats.items()
        }
        config_file.write_text(json.dumps(config))
    ref = reference_model(checkpoint)
    gpu = _rs.Lfm2Model(str(checkpoint), device, 20, max_buffer_bytes=16384)
    assert len(gpu.weight_formats()) == 12
    qualify_short(gpu, ref)


@pytest.mark.parametrize("kind", ["f32", "mlx4", "mlx6", "nvfp4", "q4_k"])
def test_embedding_and_projection_row_chunks(tmp_path, device, kind):
    checkpoint = tiny_checkpoint(
        tmp_path, hidden=256, heads=4, kvheads=2, intermediate=512, quant=kind
    )
    ref = reference_model(checkpoint)
    # The embedding must cross several chunks, including a non-power-of-two
    # row count for Q4_K. Use enough bytes for the largest single matrix row.
    gpu = _rs.Lfm2Model(str(checkpoint), device, 20, max_buffer_bytes=4099)
    qualify_short(gpu, ref)


@pytest.mark.parametrize(
    "target",
    ["f32", "f16", "bf16", "q4_0", "q4_k", "q8_0", "mlx2", "mlx4", "mlx6", "mlx8"],
)
def test_explicit_residency_against_exported_bytes(tmp_path, device, target):
    source = tiny_checkpoint(
        tmp_path / "source",
        hidden=256,
        heads=4,
        kvheads=2,
        intermediate=512,
        quant="mlx6",
        group_size=128,
        sharded=True,
    )
    exported = tmp_path / "converted"
    _rs.Lfm2Model.convert_checkpoint(str(source), str(exported), target)
    ref = reference_model(exported)
    gpu = _rs.Lfm2Model(str(source), device, 20, quant=target)
    qualify_short(gpu, ref)
    # Reload the inspectable packed artifact through the native path as well.
    native = _rs.Lfm2Model(str(exported), device, 20)
    qualify_short(native, ref)


def test_quantization_and_shard_rejections(tmp_path, device):
    checkpoint = tiny_checkpoint(
        tmp_path / "source", hidden=256, intermediate=512, mixed=True, sharded=True
    )
    for kwargs in (
        {"quant": "q8"},
        {"quant": "mlx4", "group_size": 48},
        {"max_buffer_bytes": 0},
        {"max_buffer_bytes": 1},
    ):
        with pytest.raises(ValueError):
            _rs.Lfm2Model(str(checkpoint), device, 20, **kwargs)
    index = checkpoint / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())
    first = next(iter(mapping["weight_map"]))
    mapping["weight_map"][first] = "missing.safetensors"
    index.write_text(json.dumps(mapping))
    with pytest.raises(ValueError):
        _rs.Lfm2Model(str(checkpoint), device, 20)


def test_fp8_auxiliary_shape_guards(tmp_path, device):
    from safetensors.torch import load_file, save_file

    checkpoint = tiny_checkpoint(tmp_path, hidden=256, intermediate=512, quant="fp8")
    file = checkpoint / "model.safetensors"
    weights = load_file(file)
    key = "language_model.model.embed_tokens.weight_scale"
    original = weights[key]
    weights[key] = original.reshape(1, -1)
    save_file(weights, file)
    with pytest.raises(ValueError, match="per-output-row"):
        _rs.Lfm2Model(str(checkpoint), device, 20)
    weights[key] = original.reshape(-1, 1)
    save_file(weights, file)
    gpu = _rs.Lfm2Model(str(checkpoint), device, 20)
    qualify_short(gpu, reference_model(checkpoint))


def test_fp8_global_exporter_metadata_with_dense_embedding(tmp_path, device):
    from lfm2_formats import MatrixReader
    from safetensors.torch import load_file, save_file

    checkpoint = tiny_checkpoint(tmp_path, hidden=256, intermediate=512, quant="fp8")
    root = "language_model.model.embed_tokens"
    reader = MatrixReader(checkpoint)
    dense = reader.dequant(root, 0, 128).half()
    del reader
    file = checkpoint / "model.safetensors"
    weights = load_file(file)
    weights[root + ".weight"] = dense
    del weights[root + ".weight_scale"]
    save_file(weights, file)
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
    }
    path.write_text(json.dumps(config))
    gpu = _rs.Lfm2Model(str(checkpoint), device, 20)
    assert set(gpu.weight_formats()) == {"fp8", "f16"}
    qualify_short(gpu, reference_model(checkpoint))


@pytest.mark.parametrize("kind", ["mlx4", "fp8", "nvfp4", "f32"])
def test_per_matrix_format_metadata_must_match_storage(tmp_path, device, kind):
    checkpoint = tiny_checkpoint(tmp_path, hidden=256, intermediate=512, quant=kind)
    path = checkpoint / "config.json"
    config = json.loads(path.read_text())
    config.pop("quantization", None)
    config["quantization_config"] = {
        "tensor_formats": {
            "model.embed_tokens": {
                "format": "bad_format",
                "mode": "affine",
                "bits": 4,
                "group_size": 64,
            }
        }
    }
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="format metadata"):
        _rs.Lfm2Model(str(checkpoint), device, 20)
