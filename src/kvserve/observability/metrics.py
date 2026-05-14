from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from prometheus_client import Counter, Gauge, Histogram, REGISTRY, generate_latest


@dataclass(slots=True)
class Metrics:
    kv_memory_gpu_bytes: Gauge
    kv_memory_cpu_bytes: Gauge
    kv_memory_nvme_bytes: Gauge
    kv_compression_ratio: Gauge
    kv_prefix_reuse_rate: Gauge
    prefix_reuse_lookups: Counter
    prefix_reuse_hits: Counter
    kv_pruned_tokens: Counter
    kv_evictions: Counter
    active_policy: Gauge
    quantization_mode_active: Gauge
    request_latency: Histogram
    tokens_generated: Counter
    requests_total: Counter


def create_metrics() -> Metrics:
    return Metrics(
        kv_memory_gpu_bytes=_gauge("kv_memory_gpu_bytes", "Resident GPU KV bytes"),
        kv_memory_cpu_bytes=_gauge("kv_memory_cpu_bytes", "Resident CPU KV bytes"),
        kv_memory_nvme_bytes=_gauge("kv_memory_nvme_bytes", "Resident NVMe KV bytes"),
        kv_compression_ratio=_gauge(
            "kv_compression_ratio", "Original KV bytes divided by compressed KV bytes", ["model_id"]
        ),
        kv_prefix_reuse_rate=_gauge("kv_prefix_reuse_rate", "Prefix KV reuse hit rate"),
        prefix_reuse_lookups=_counter("kv_prefix_reuse_lookups_total", "KV prefix reuse lookups"),
        prefix_reuse_hits=_counter("kv_prefix_reuse_hits_total", "KV prefix reuse hits"),
        kv_pruned_tokens=_counter("kv_pruned_tokens_total", "KV tokens pruned by policy"),
        kv_evictions=_counter("kv_evictions_total", "KV pages evicted or demoted"),
        active_policy=_gauge("kv_policy_active", "Active KV policy decision", ["model_id", "policy"]),
        quantization_mode_active=_gauge(
            "quantization_mode_active", "Active quantization mode", ["model_id", "mode"]
        ),
        request_latency=_histogram(
            "request_latency_seconds",
            "Request latency by route",
            ["route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
        ),
        tokens_generated=_counter("tokens_generated_total", "Generated tokens", ["model_id"]),
        requests_total=_counter("requests_total", "Requests by route/status", ["route", "status"]),
    )


def render_metrics() -> bytes:
    return generate_latest(REGISTRY)


class latency_timer:
    def __init__(self, metrics: Metrics, route: str):
        self.metrics = metrics
        self.route = route
        self.start = 0.0

    def __enter__(self) -> "latency_timer":
        self.start = perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.metrics.request_latency.labels(route=self.route).observe(perf_counter() - self.start)


def _counter(name: str, documentation: str, labelnames: list[str] | None = None) -> Counter:
    existing = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
    if existing:
        return existing  # type: ignore[return-value]
    return Counter(name, documentation, labelnames or [])


def _gauge(name: str, documentation: str, labelnames: list[str] | None = None) -> Gauge:
    existing = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
    if existing:
        return existing  # type: ignore[return-value]
    return Gauge(name, documentation, labelnames or [])


def _histogram(
    name: str, documentation: str, labelnames: list[str] | None = None, buckets: tuple[float, ...] = ()
) -> Histogram:
    existing = REGISTRY._names_to_collectors.get(name)  # type: ignore[attr-defined]
    if existing:
        return existing  # type: ignore[return-value]
    return Histogram(name, documentation, labelnames or [], buckets=buckets)
