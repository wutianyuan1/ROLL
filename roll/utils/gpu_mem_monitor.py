import time
from contextlib import ContextDecorator
import pynvml


class GPUMemoryMonitor(ContextDecorator):
    """
    Context manager monitors all GPU's memory usage on this node
    """
    def __init__(self, interval_sec: float = None, print_fn=print):
        """
        :param interval_sec: interval of gpu memory usage sampling
        :param print_fn: logging function used to print output
        """
        self.interval_sec = interval_sec
        self.print_fn = print_fn
        self._running = False
        pynvml

    def __enter__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.print_fn(f"[GPUMonitor] Total #GPUs: {self.device_count}")

        self.print_fn("[GPUMonitor] On enter: ")
        self._log_memory()

        if self.interval_sec and self.interval_sec > 0:
            self._running = True
            import threading
            self.thread = threading.Thread(target=self._loop_monitor, daemon=True)
            self.thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1)
        self.print_fn("[GPUMonitor] On exit: ")
        self._log_memory()
        pynvml.nvmlShutdown()

    def _log_memory(self):
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = mem_info.used / 1024**2
            total_mb = mem_info.total / 1024**2
            self.print_fn(f"  GPU {i}: {used_mb:.1f} MB / {total_mb:.1f} MB")

    def _loop_monitor(self):
        while self._running:
            self.print_fn("[GPUMonitor] GPU memory used: ")
            self._log_memory()
            time.sleep(self.interval_sec)
