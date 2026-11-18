"""Append-only local measurements reusable by future Run/Trial/Tool runtimes."""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class EventLog:
    def __init__(self, path: Path, context: dict):
        self.path, self.context = path, context
        self.started = time.perf_counter()

    def emit(self, event: str, details: dict | None = None):
        record = {**self.context, "time_utc": utc_now(), "elapsed_seconds": time.perf_counter() - self.started,
                  "event": event, "details": details or {}}
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


class ResourceMonitor:
    """Sample current-process RSS/CPU; not a hard memory limiter or GPU monitor."""

    def __init__(self, path: Path, interval=0.1):
        self.path, self.interval = path, interval
        self.stop = threading.Event()
        self.thread = None
        self.peak_rss_bytes = 0

    def sample(self, process):
        rss = process.memory_info().rss
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        record = {"time_utc": utc_now(), "pid": process.pid, "rss_bytes": rss,
                  "process_cpu_percent": process.cpu_percent(None), "host_cpu_percent": psutil.cpu_percent(None),
                  "host_available_memory_bytes": psutil.virtual_memory().available, "gpu_utilization": None}
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")

    def __enter__(self):
        process = psutil.Process()
        self.sample(process)

        def watch():
            while not self.stop.wait(self.interval):
                self.sample(process)

        self.thread = threading.Thread(target=watch, name="resource-monitor", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exception):
        self.stop.set()
        self.thread.join()
        self.sample(psutil.Process())
