#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny SM121 sparse-indexer fallback smoke.

Run only in an isolated GB10 test container or lease window, not inside a live
serving pod:

    VLLM_SM121_SPARSE_INDEXER_TORCH_FALLBACK=1 \
      python tools/sm121_sparse_indexer_smoke.py
"""

import os
import sys

import torch

from vllm.v1.attention.ops.sm121_sparse_indexer import (
    fp8_mqa_logits_torch,
    fp8_paged_mqa_logits_torch,
    sm121_sparse_indexer_torch_fallback_enabled,
)


def _require_sm121() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(f"expected SM121 / capability (12, 1), got {capability}")
    if os.environ.get("VLLM_SM121_SPARSE_INDEXER_TORCH_FALLBACK") != "1":
        raise RuntimeError("set VLLM_SM121_SPARSE_INDEXER_TORCH_FALLBACK=1")
    if not sm121_sparse_indexer_torch_fallback_enabled():
        raise RuntimeError("fallback gate did not enable")


def _check_unpaged(device: torch.device) -> None:
    q = torch.tensor(
        [[[1.0, 2.0], [0.5, -1.0]], [[-1.0, 1.0], [2.0, 1.0]]],
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    k = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    scales = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32, device=device)
    weights = torch.tensor(
        [[0.25, 0.75], [1.0, 0.5]], dtype=torch.float32, device=device
    )
    starts = torch.tensor([0, 1], dtype=torch.int32, device=device)
    ends = torch.tensor([3, 3], dtype=torch.int32, device=device)

    got = fp8_mqa_logits_torch(q, (k, scales), weights, starts, ends)
    score = (
        torch.einsum("mhd,nd->hmn", q.to(torch.bfloat16), k.to(torch.bfloat16))
        .float()
        .mul(scales.view(1, 1, -1))
    )
    expected = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    mask = torch.tensor(
        [[True, True, True], [False, True, True]], dtype=torch.bool, device=device
    )
    expected = expected.masked_fill(~mask, float("-inf"))
    torch.testing.assert_close(got, expected)


def _check_paged(device: torch.device) -> None:
    block_size = 2
    dim = 2
    q = torch.tensor(
        [[[[1.0, 2.0], [0.5, -1.0]]]], dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    values = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0]], [[1.0, 1.0], [2.0, 0.0]]],
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1.0, 0.5], [2.0, 1.5]], dtype=torch.float32,
                          device=device)

    kv_cache = torch.empty((2, block_size, 1, dim + 4), dtype=torch.uint8,
                           device=device)
    flat = kv_cache.view(2, block_size * (dim + 4))
    flat[:, : block_size * dim] = values.view(torch.uint8).view(
        2, block_size * dim
    )
    flat[:, block_size * dim :] = scales.view(torch.uint8).view(
        2, block_size * 4
    )
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32, device=device)
    context_lens = torch.tensor([3], dtype=torch.int32, device=device)
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32, device=device)

    got = fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len=4
    )
    cache_values = values.reshape(4, dim).to(torch.float32)
    cache_scales = scales.reshape(4)
    score = torch.nn.functional.linear(cache_values[:3], q[0, 0].to(torch.float32))
    expected = torch.full((1, 4), float("-inf"), dtype=torch.float32,
                          device=device)
    expected[0, :3] = (score.relu() * weights.view(1, -1)).sum(dim=1) * (
        cache_scales[:3]
    )
    torch.testing.assert_close(got, expected)


def main() -> int:
    _require_sm121()
    device = torch.device("cuda")
    _check_unpaged(device)
    _check_paged(device)
    torch.cuda.synchronize()
    print("SM121 sparse-indexer fallback smoke OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SM121 sparse-indexer fallback smoke FAILED: {exc}", file=sys.stderr)
        raise
