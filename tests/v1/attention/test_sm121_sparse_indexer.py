# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.ops.sm121_sparse_indexer import (
    fp8_mqa_logits_torch,
    fp8_paged_mqa_logits_torch,
)


def test_fp8_mqa_logits_torch_matches_reference():
    q = torch.tensor(
        [[[1.0, 2.0], [0.5, -1.0]], [[-1.0, 1.0], [2.0, 1.0]]],
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    k = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    scales = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32)
    weights = torch.tensor([[0.25, 0.75], [1.0, 0.5]], dtype=torch.float32)
    starts = torch.tensor([0, 1], dtype=torch.int32)
    ends = torch.tensor([3, 3], dtype=torch.int32)

    got = fp8_mqa_logits_torch(q, (k, scales), weights, starts, ends)

    q_ref = q.to(torch.bfloat16)
    k_ref = k.to(torch.bfloat16)
    score = torch.einsum("mhd,nd->hmn", q_ref, k_ref).float()
    score *= scales.view(1, 1, -1)
    expected = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    mask = torch.tensor([[True, True, True], [False, True, True]])
    expected = expected.masked_fill(~mask, float("-inf"))

    torch.testing.assert_close(got, expected)


def test_fp8_paged_mqa_logits_torch_next_n_one_matches_reference():
    block_size = 2
    dim = 2
    q = torch.tensor(
        [[[[1.0, 2.0], [0.5, -1.0]]]],
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    values = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0]], [[1.0, 1.0], [2.0, 0.0]]],
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1.0, 0.5], [2.0, 1.5]], dtype=torch.float32)

    kv_cache = torch.empty((2, block_size, 1, dim + 4), dtype=torch.uint8)
    kv_cache_flat = kv_cache.view(2, block_size * (dim + 4))
    kv_cache_flat[:, : block_size * dim] = values.view(torch.uint8).view(
        2, block_size * dim
    )
    kv_cache_flat[:, block_size * dim :] = scales.view(torch.uint8).view(
        2, block_size * 4
    )

    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    context_lens = torch.tensor([3], dtype=torch.int32)
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)

    got = fp8_paged_mqa_logits_torch(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len=4,
    )

    cache_values = values.reshape(4, dim).to(torch.float32)
    cache_scales = scales.reshape(4)
    score = torch.nn.functional.linear(cache_values[:3], q[0, 0].to(torch.float32))
    expected = torch.full((1, 4), float("-inf"), dtype=torch.float32)
    expected[0, :3] = (score.relu() * weights.view(1, -1)).sum(dim=1) * (
        cache_scales[:3]
    )

    torch.testing.assert_close(got, expected)
