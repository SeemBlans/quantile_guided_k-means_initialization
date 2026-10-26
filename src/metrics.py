import os, threading
import psutil
import numpy as np
from sklearn.metrics import silhouette_score


class PeakRSSSampler:
    """Sample process RSS during a measured run.

    baseline_bytes: RSS measured when sampling begins.
    peak_bytes: maximum sampled RSS, including baseline.
    peak_increase_bytes: peak RSS above baseline.

    Measurements are approximate: short-lived peaks may be missed,
    and the sampling thread itself adds a small amount of overhead.
    """
    def __init__(self, interval=0.01):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self.stop = threading.Event()
        self.thread = None

    def _sample(self):
        while not self.stop.is_set():
            try:
                rss = self.proc.memory_info().rss
                self.peak_bytes = max(self.peak_bytes, rss)
            except (psutil.Error, OSError):
                pass
            self.stop.wait(self.interval)

    def __enter__(self):
        try:
            self.baseline_bytes = self.proc.memory_info().rss
            self.peak_bytes = self.baseline_bytes
        except (psutil.Error, OSError):
            self.baseline_bytes = 0
            self.peak_bytes = 0

        self.stop.clear()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

        try:
            self.peak_bytes = max(
                self.peak_bytes,
                self.proc.memory_info().rss
            )
        except (psutil.Error, OSError):
            pass

    @property
    def peak_increase_bytes(self):
        """Approximate peak RSS increase above the starting baseline."""
        return max(0, self.peak_bytes - self.baseline_bytes)


def calculate_silhouette(X_sample, labels_sample):
    """Calculate silhouette on a fixed, shared evaluation sample."""
    labels_sample = np.asarray(labels_sample)
    unique = np.unique(labels_sample)

    if len(unique) < 2 or len(unique) >= len(labels_sample):
        return float("nan")

    return float(
        silhouette_score(
            X_sample,
            labels_sample,
            metric="euclidean"
        )
    )
