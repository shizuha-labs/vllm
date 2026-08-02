# SPDX-License-Identifier: Apache-2.0
"""Expose RFC-37003 retention state through vLLM Prometheus metrics."""

from pathlib import Path

_base = Path(__file__).with_name("loggers_base.py")
exec(compile(_base.read_bytes(), str(_base), "exec"), globals(), globals())

_BasePrometheusStatLogger = PrometheusStatLogger


class PrometheusStatLogger(_BasePrometheusStatLogger):
    _RETENTION_GAUGES = (
        "protected_blocks",
        "queued_protected_blocks",
        "budget_blocks",
    )
    _RETENTION_COUNTERS = (
        "priority_evictions_total",
        "ttl_expiries_total",
        "budget_drops_total",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labelnames = ["model_name", "engine"]
        values = self.per_engine_labelvalues
        self.retention_gauges = {}
        for key in self._RETENTION_GAUGES:
            metric = self._gauge_cls(
                name=f"vllm:retention_{key}",
                documentation=f"KV retention {key.replace('_', ' ')}.",
                multiprocess_mode="mostrecent",
                labelnames=labelnames,
            )
            self.retention_gauges[key] = create_metric_per_engine(metric, values)

        self.retention_counters = {}
        for key in self._RETENTION_COUNTERS:
            metric = self._counter_cls(
                name=f"vllm:retention_{key}",
                documentation=f"KV retention {key.replace('_', ' ')}.",
                labelnames=labelnames,
            )
            self.retention_counters[key] = create_metric_per_engine(metric, values)
        self._last_retention_counters = {
            engine_idx: {key: 0 for key in self._RETENTION_COUNTERS}
            for engine_idx in self.engine_indexes
        }

    def record(self, scheduler_stats, *args, engine_idx=0, **kwargs):
        super().record(scheduler_stats, *args, engine_idx=engine_idx, **kwargs)
        if scheduler_stats is None:
            return
        metrics = getattr(scheduler_stats, "retention_metrics", {})
        for key, metric_by_engine in self.retention_gauges.items():
            metric_by_engine[engine_idx].set(metrics.get(key, 0))
        previous = self._last_retention_counters[engine_idx]
        for key, metric_by_engine in self.retention_counters.items():
            current = metrics.get(key, 0)
            delta = current - previous[key]
            metric_by_engine[engine_idx].inc(current if delta < 0 else delta)
            previous[key] = current
