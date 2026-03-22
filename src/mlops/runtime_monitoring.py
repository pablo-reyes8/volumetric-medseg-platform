import statistics
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import psutil
import torch

from src.api.settings import Settings


@dataclass
class RequestEvent:
    timestamp: float
    path: str
    method: str
    status_code: int
    latency_ms: float


class RuntimeMonitor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.started_at = time.time()
        self._events: Deque[RequestEvent] = deque()
        self._lock = threading.Lock()
        self._process = psutil.Process()

    def record_request(self, path: str, method: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self._events.append(RequestEvent(time.time(), path, method, status_code, latency_ms))
            self._trim_locked()

    def _trim_locked(self) -> None:
        cutoff = time.time() - self.settings.monitoring_window_seconds
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

    def _snapshot_events(self) -> List[RequestEvent]:
        with self._lock:
            self._trim_locked()
            return list(self._events)

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        rank = (len(values) - 1) * percentile
        low_index = int(rank)
        high_index = min(low_index + 1, len(values) - 1)
        fraction = rank - low_index
        ordered = sorted(values)
        return ordered[low_index] + (ordered[high_index] - ordered[low_index]) * fraction

    def snapshot(self) -> Dict[str, object]:
        events = self._snapshot_events()
        latencies = [event.latency_ms for event in events]
        errors = [event for event in events if event.status_code >= 400]
        uptime_seconds = time.time() - self.started_at
        requests = len(events)
        requests_per_minute = requests / max(1e-6, min(uptime_seconds, self.settings.monitoring_window_seconds) / 60.0)

        endpoint_counter: Dict[str, Dict[str, object]] = {}
        grouped: Dict[str, List[RequestEvent]] = defaultdict(list)
        for event in events:
            grouped[f"{event.method} {event.path}"].append(event)
        for key, endpoint_events in grouped.items():
            endpoint_latencies = [event.latency_ms for event in endpoint_events]
            endpoint_errors = sum(1 for event in endpoint_events if event.status_code >= 400)
            endpoint_counter[key] = {
                "requests": len(endpoint_events),
                "error_rate": round(endpoint_errors / max(1, len(endpoint_events)), 6),
                "p95_latency_ms": round(self._percentile(endpoint_latencies, 0.95), 3),
            }

        cpu_percent = self._process.cpu_percent(interval=None)
        memory_info = self._process.memory_info()
        system_memory = psutil.virtual_memory()
        gpu_memory_reserved_mb = 0.0
        gpu_memory_allocated_mb = 0.0
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_memory_reserved_mb = round(torch.cuda.memory_reserved() / (1024 ** 2), 3)
            gpu_memory_allocated_mb = round(torch.cuda.memory_allocated() / (1024 ** 2), 3)

        hourly_rate = self.settings.gpu_hourly_cost_usd if gpu_available else self.settings.cpu_hourly_cost_usd
        memory_hourly_cost = (memory_info.rss / (1024 ** 3)) * self.settings.memory_gb_hourly_cost_usd
        total_hourly_rate = hourly_rate + memory_hourly_cost
        total_estimated_cost = (uptime_seconds / 3600.0) * total_hourly_rate
        cost_per_1000_requests = (total_estimated_cost / requests * 1000.0) if requests else 0.0

        return {
            "window_seconds": self.settings.monitoring_window_seconds,
            "uptime_seconds": round(uptime_seconds, 3),
            "totals": {
                "requests": requests,
                "errors": len(errors),
                "error_rate": round(len(errors) / max(1, requests), 6),
            },
            "throughput": {
                "requests_per_minute": round(requests_per_minute, 3),
            },
            "latency_ms": {
                "avg": round(statistics.mean(latencies), 3) if latencies else 0.0,
                "p50": round(self._percentile(latencies, 0.50), 3),
                "p95": round(self._percentile(latencies, 0.95), 3),
                "max": round(max(latencies), 3) if latencies else 0.0,
            },
            "resources": {
                "cpu_percent": round(cpu_percent, 3),
                "memory_rss_mb": round(memory_info.rss / (1024 ** 2), 3),
                "memory_percent": round(system_memory.percent, 3),
                "gpu_available": gpu_available,
                "gpu_memory_reserved_mb": gpu_memory_reserved_mb,
                "gpu_memory_allocated_mb": gpu_memory_allocated_mb,
            },
            "cost_estimate": {
                "cpu_hourly_cost_usd": self.settings.cpu_hourly_cost_usd,
                "gpu_hourly_cost_usd": self.settings.gpu_hourly_cost_usd,
                "memory_gb_hourly_cost_usd": self.settings.memory_gb_hourly_cost_usd,
                "estimated_total_cost_usd": round(total_estimated_cost, 6),
                "estimated_cost_per_1000_requests_usd": round(cost_per_1000_requests, 6),
            },
            "endpoints": endpoint_counter,
        }
