# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-first sparse-indexer fallback for GB10 / SM121.

This is intentionally opt-in. It gives SM121 a non-DeepGEMM CUDA path for the
DeepSeek/GLM sparse-attention indexer logits, using torch ops. It is a
compatibility foothold and reference implementation for a future Triton/CUDA
kernel, not the final high-throughput production path.
"""

import os

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv


def sm121_sparse_indexer_torch_fallback_enabled() -> bool:
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return (
        capability is not None
        and capability.major == 12
        and capability.minor == 1
        and os.environ.get("VLLM_SM121_SPARSE_INDEXER_TORCH_FALLBACK") == "1"
    )


def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single unpaged sequence.

    Reference formula matches the ROCm torch fallback, which was copied from
    DeepGEMM tests. Shapes:
      q: [M, H, D] float8_e4m3fn
      kv[0]: [N, D] float8_e4m3fn
      kv[1]: [N] or [N, 1] float32 K scales
      weights: [M, H] float32
    """
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    device = q.device
    k = k_fp8.to(torch.bfloat16)
    q = q.to(torch.bfloat16)
    scale = scale.view(-1).to(torch.float32)

    offsets = torch.arange(seq_len_kv, device=device)
    mask = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
        offsets[None, :] < cu_seqlen_ke[:, None]
    )

    score = torch.einsum("mhd,nd->hmn", q, k).float()
    score = score * scale.view(1, 1, -1)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    return logits.masked_fill(~mask, float("-inf"))


def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a paged KV cache.

    This supports the FP8 cache layout used by GLM-5.x sparse indexer:
    [num_blocks, block_size, 1, D + 4] uint8 where the final four bytes per
    token store a float32 dequant scale.
    """
    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    block_size = kv_cache.shape[1]
    device = q.device

    if next_n == 1:
        logits = torch.full(
            (batch_size, max_model_len),
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )
        if context_lens.dim() > 1:
            context_lens = context_lens.squeeze(-1)
        kv_cache_flat = kv_cache.view(-1, block_size * (dim + 4))
        scale_offset = block_size * dim
        for i in range(batch_size):
            q_i = q[i, 0].to(torch.float32)
            q_scale = weights[i].to(torch.float32)
            seq_len = int(context_lens[i].item())
            num_pages = cdiv(seq_len, block_size)
            padded_seq_len = num_pages * block_size
            pages = block_tables[i, :num_pages]
            cache = kv_cache_flat[pages]
            cache_value = cache[..., :scale_offset].view(dtype=fp8_dtype).to(
                torch.float32
            )
            cache_scale = cache[..., scale_offset:].view(dtype=torch.float32)
            cache_value = cache_value.view(padded_seq_len, dim)
            cache_scale = cache_scale.contiguous().view(padded_seq_len)
            score = F.linear(cache_value, q_i).relu()
            score = (score * q_scale.view(1, -1)).sum(dim=1)
            score = score * cache_scale
            logits[i, :seq_len] = score[:seq_len]
        return logits

    kv_cache_flat = kv_cache.view(-1, block_size * (dim + 4))
    scale_offset = block_size * dim
    q = q.float()
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )

    for i in range(batch_size):
        context_len = context_lens[i]
        if context_len.ndim == 0:
            context_len_i = int(context_len.item())
            q_offsets = torch.arange(
                context_len_i - next_n, context_len_i, device=device
            )
            context_limit = torch.full(
                (next_n,), context_len_i, dtype=torch.int32, device=device
            )
        else:
            context_limit = context_len.to(device=device, dtype=torch.int32)
            q_offsets = context_limit - 1

        weight_slice = (
            weights[i * next_n : (i + 1) * next_n].transpose(0, 1).contiguous()
        )
        max_context_len = int(context_limit.max().item())
        for block_rk in range(cdiv(max_context_len, block_size)):
            block_idx = block_tables[i, block_rk]
            qx = q[i]
            cache = kv_cache_flat[block_idx]
            kx = cache[:scale_offset].view(dtype=fp8_dtype).to(torch.float32)
            kx = kx.view(block_size, dim)
            k_scale = cache[scale_offset:].view(dtype=torch.float32).view(block_size)
            kx = kx * k_scale[:, None]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device=device
            )
            mask = (k_offsets[None, :] < context_limit[:, None]) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            score = qx.transpose(0, 1) @ kx.transpose(0, 1)
            score = torch.where(mask[None, :, :], score.float(), float("-inf"))
            score = (score.relu() * weight_slice[..., None]).sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], score,
                            float("-inf"))

    return logits
