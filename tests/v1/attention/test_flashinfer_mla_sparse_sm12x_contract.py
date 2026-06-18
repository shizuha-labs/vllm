# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseBackend,
)


def test_sm12x_flashinfer_sparse_requires_quantized_kv_cache(monkeypatch):
    import vllm.config as config_module

    fake_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(qk_nope_head_dim=128, index_topk=2048)
        )
    )
    monkeypatch.setattr(config_module, "get_current_vllm_config", lambda: fake_config)

    reason = FlashInferMLASparseBackend.supports_combination(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=64,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        device_capability=DeviceCapability(12, 1),
    )
    assert reason is not None
    assert "requires an fp8 KV cache" in reason

    reason = FlashInferMLASparseBackend.supports_combination(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="fp8",
        block_size=64,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        device_capability=DeviceCapability(12, 1),
    )
    assert reason is None


def test_sm12x_flashinfer_sparse_fp8_ds_mla_cache_shape():
    assert FlashInferMLASparseBackend.supports_compute_capability(
        DeviceCapability(12, 1)
    )
    assert FlashInferMLASparseBackend.get_kv_cache_shape(
        num_blocks=4,
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        cache_dtype_str="fp8_ds_mla",
    ) == (4, 64, 656)
