#!/usr/bin/env python3
"""
Robust GPU memory and runtime tracking utilities.

Prevents silent failures - if GPU tracking fails, raises clear errors instead of returning 0.
Uses pynvml for accurate driver-level GPU memory tracking (same as nvidia-smi).
"""
import torch
import time
import warnings
from typing import Optional, Dict, Any

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not available - GPU memory tracking will be less accurate. Install with: pip install pynvml")


class GPUTracker:
    """Track GPU memory and runtime with robust error handling."""

    def __init__(self, device: torch.device, require_gpu: bool = False):
        """
        Initialize GPU tracker.

        Args:
            device: PyTorch device to track
            require_gpu: If True, raises error if CUDA not available
        """
        self.device = device
        self.is_cuda = torch.cuda.is_available() and device.type == "cuda"

        if require_gpu and not self.is_cuda:
            raise RuntimeError(
                f"GPU required but not available! Device: {device}, "
                f"CUDA available: {torch.cuda.is_available()}"
            )

        self.start_time: Optional[float] = None
        self.baseline_memory_bytes: Optional[int] = None
        self.nvml_handle = None
        self.nvml_initialized = False

        # Initialize NVML for accurate GPU tracking
        if self.is_cuda and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                gpu_index = device.index if device.index is not None else 0
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self.nvml_initialized = True
            except Exception as e:
                warnings.warn(f"Failed to initialize NVML: {e}. Falling back to PyTorch tracking.")

        # Log GPU info at initialization
        if self.is_cuda:
            self._log_gpu_info()

    def _log_gpu_info(self):
        """Log GPU information for debugging."""
        try:
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_mem_total = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            print(f"[GPU Tracker] Using: {gpu_name}")
            print(f"[GPU Tracker] Total memory: {gpu_mem_total:.2f} GB")
        except Exception as e:
            warnings.warn(f"Could not retrieve GPU info: {e}")

    def start(self):
        """Start tracking time and record baseline GPU memory."""
        self.start_time = time.time()

        if self.is_cuda:
            try:
                # Record baseline memory from driver (NVML)
                if self.nvml_initialized:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    self.baseline_memory_bytes = mem_info.used

                # Also reset PyTorch stats for fallback
                torch.cuda.reset_peak_memory_stats(self.device)
                # Synchronize to ensure accurate timing
                torch.cuda.synchronize(self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to reset GPU memory stats: {e}")

    def stop(self) -> Dict[str, Any]:
        """
        Stop tracking and return metrics.

        Returns:
            Dictionary with:
                - runtime_sec: Training/evaluation time in seconds
                - peak_memory_mb: Peak GPU memory in MB (or -1 if no GPU)
                - peak_memory_gb: Peak GPU memory in GB (or -1 if no GPU)

        Raises:
            RuntimeError: If tracking was not started or GPU stats retrieval fails
        """
        if self.start_time is None:
            raise RuntimeError("Tracker was not started! Call .start() first")

        # Get runtime
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        runtime_sec = time.time() - self.start_time

        # Get peak memory
        if self.is_cuda:
            try:
                # Use NVML for accurate driver-level tracking (like nvidia-smi)
                if self.nvml_initialized and self.baseline_memory_bytes is not None:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    current_memory_bytes = mem_info.used
                    # Calculate memory used by this process (delta from baseline)
                    peak_memory_bytes = current_memory_bytes - self.baseline_memory_bytes

                    # If negative (memory freed), use absolute value as peak was during training
                    # This can happen if baseline had temp allocations
                    if peak_memory_bytes < 0:
                        peak_memory_bytes = abs(peak_memory_bytes)

                    print(f"[GPU Tracker] Driver-level memory used: {peak_memory_bytes / (1024**2):.2f} MB (NVML)")

                else:
                    # Fallback to PyTorch tracking (less accurate)
                    peak_memory_bytes = torch.cuda.max_memory_allocated(self.device)
                    print(f"[GPU Tracker] PyTorch-level memory used: {peak_memory_bytes / (1024**2):.2f} MB (less accurate)")

                    # Sanity check for PyTorch tracking only
                    if peak_memory_bytes < 10 * 1024 * 1024:  # < 10 MB
                        warnings.warn(
                            f"⚠️  PyTorch tracked memory is low: {peak_memory_bytes / (1024**2):.2f} MB. "
                            "This may be inaccurate with precomputed embeddings. Install pynvml for accurate tracking: pip install pynvml"
                        )

                peak_memory_mb = peak_memory_bytes / (1024 * 1024)
                peak_memory_gb = peak_memory_bytes / (1024 * 1024 * 1024)

            except Exception as e:
                raise RuntimeError(f"Failed to retrieve GPU memory stats: {e}")
        else:
            # No GPU - use sentinel value -1 (not 0, to distinguish from failed tracking)
            peak_memory_mb = -1.0
            peak_memory_gb = -1.0

        # Reset for next use
        self.start_time = None

        return {
            "runtime_sec": runtime_sec,
            "peak_memory_mb": peak_memory_mb,
            "peak_memory_gb": peak_memory_gb,
        }

    def get_current_memory_mb(self) -> float:
        """Get current GPU memory usage in MB (or -1 if no GPU)."""
        if self.is_cuda:
            try:
                return torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            except Exception as e:
                raise RuntimeError(f"Failed to get current GPU memory: {e}")
        return -1.0

    def __del__(self):
        """Clean up NVML on destruction."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass  # Ignore errors during cleanup


def log_gpu_summary(metrics: Dict[str, Any], prefix: str = ""):
    """
    Pretty-print GPU tracking results.

    Args:
        metrics: Dictionary from GPUTracker.stop()
        prefix: Optional prefix for log messages
    """
    runtime_sec = metrics["runtime_sec"]
    peak_memory_mb = metrics["peak_memory_mb"]
    peak_memory_gb = metrics["peak_memory_gb"]

    print(f"\n{'='*60}")
    print(f"{prefix}Performance Metrics")
    print(f"{'='*60}")
    print(f"Runtime: {runtime_sec:.2f}s ({runtime_sec/60:.2f} min)")

    if peak_memory_mb > 0:
        print(f"Peak GPU Memory: {peak_memory_mb:.2f} MB ({peak_memory_gb:.4f} GB)")
    else:
        print(f"Peak GPU Memory: Not tracked (CPU execution)")

    print(f"{'='*60}\n")


# Example usage:
if __name__ == "__main__":
    # Test with GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = GPUTracker(device)

    tracker.start()

    # Simulate some work
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device=device)
        y = x @ x.T
        time.sleep(0.5)

    metrics = tracker.stop()
    log_gpu_summary(metrics, prefix="Test ")

    # Print raw metrics
    print("Raw metrics:", metrics)
