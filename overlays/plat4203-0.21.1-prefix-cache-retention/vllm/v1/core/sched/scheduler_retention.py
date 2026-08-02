# SPDX-License-Identifier: Apache-2.0
"""Add retention telemetry and bounded APC replay to the vendor scheduler."""

import os
from pathlib import Path

_base = Path(__file__).with_name("scheduler_base.py")
exec(compile(_base.read_bytes(), str(_base), "exec"), globals(), globals())

_BaseScheduler = Scheduler


def _prefix_cache_min_recompute_tokens() -> int:
    raw = os.environ.get("VLLM_PREFIX_CACHE_MIN_RECOMPUTE_TOKENS", "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            "VLLM_PREFIX_CACHE_MIN_RECOMPUTE_TOKENS must be an integer"
        ) from exc
    if value < 0:
        raise RuntimeError(
            "VLLM_PREFIX_CACHE_MIN_RECOMPUTE_TOKENS must be non-negative"
        )
    return value


def _install_prefix_cache_recompute_floor(manager, min_recompute_tokens: int) -> None:
    """Bound APC reuse so continuation prefills retain a fast query shape.

    DeepSeek V4 sparse MLA on SM121 has a severe small-continuation cliff: a
    50K-60K cached prefix with only 164-868 new tokens took 7.7-12.0 seconds,
    while replaying 1.1K-2.0K tokens took 0.67-1.76 seconds. Limit the longest
    cache hit before allocation instead of padding or mutating input tokens.
    The coordinator rounds the result to a replayable hybrid-cache boundary.
    """
    if min_recompute_tokens <= 0:
        return

    def get_computed_blocks(request):
        if not manager.enable_caching or request.skip_reading_prefix_cache:
            return manager.empty_kv_cache_blocks, 0

        max_cache_hit_length = max(0, request.num_tokens - min_recompute_tokens)
        computed_blocks, num_new_computed_tokens = (
            manager.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        if manager.log_stats:
            assert manager.prefix_cache_stats is not None
            manager.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return (
            manager.create_kv_cache_blocks(computed_blocks),
            num_new_computed_tokens,
        )

    manager.get_computed_blocks = get_computed_blocks


class Scheduler(_BaseScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _install_prefix_cache_recompute_floor(
            self.kv_cache_manager,
            _prefix_cache_min_recompute_tokens(),
        )

    def make_stats(self, *args, **kwargs):
        stats = super().make_stats(*args, **kwargs)
        if stats is not None:
            stats.retention_metrics = (
                self.kv_cache_manager.block_pool.get_retention_metrics()
            )
        return stats
