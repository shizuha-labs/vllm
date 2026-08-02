# SPDX-License-Identifier: Apache-2.0
"""Add retention telemetry to the exact vendor scheduler implementation."""

from pathlib import Path

_base = Path(__file__).with_name("scheduler_base.py")
exec(compile(_base.read_bytes(), str(_base), "exec"), globals(), globals())

_BaseScheduler = Scheduler


class Scheduler(_BaseScheduler):
    def make_stats(self, *args, **kwargs):
        stats = super().make_stats(*args, **kwargs)
        if stats is not None:
            stats.retention_metrics = (
                self.kv_cache_manager.block_pool.get_retention_metrics()
            )
        return stats
