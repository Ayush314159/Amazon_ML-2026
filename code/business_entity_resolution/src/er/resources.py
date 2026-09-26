"""Wall-time and memory tracking for experiments on a RAM-constrained machine."""
import threading
import time

import psutil


class Tracker:
    """Context manager recording elapsed seconds and peak RSS (sampled) of this process.

    >>> with Tracker() as t: ...
    >>> t.seconds, t.peak_rss_gb, t.rss_delta_gb
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._proc = psutil.Process()
        self._stop = threading.Event()

    def _sample(self):
        while not self._stop.is_set():
            self._peak = max(self._peak, self._proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self._start_rss = self._proc.memory_info().rss
        self._peak = self._start_rss
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join()
        end = self._proc.memory_info().rss
        self._peak = max(self._peak, end)
        self.start_rss_gb = self._start_rss / 1e9
        self.peak_rss_gb = self._peak / 1e9
        self.rss_delta_gb = (end - self._start_rss) / 1e9
        return False


def available_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9
