# SPDX-License-Identifier: Apache-2.0
"""Add retention fields without replacing the vendor stats implementation."""

from pathlib import Path

_base = Path(__file__).with_name("stats_base.py")
exec(compile(_base.read_bytes(), str(_base), "exec"), globals(), globals())

_BaseSchedulerStats = SchedulerStats


@dataclass
class SchedulerStats(_BaseSchedulerStats):
    retention_metrics: dict[str, int] = field(default_factory=dict)
