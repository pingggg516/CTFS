
import torch
import time
import logging
import psutil
import os
import pynvml

class PerformanceMonitor:
    """
    Monitor and record training performance metrics including:
    - Training time
    - Inference speed (FPS)
    - GPU memory usage (Peak/Current)
    - System memory usage
    """
    def __init__(self, logger=None):
        self.logger = logger
        self.start_time = None
        self.epoch_start_time = None
        self.inference_times = []
        
        # Initialize NVML for GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Monitor GPU 0 by default
            self.has_gpu_monitor = True
        except Exception:
            self.has_gpu_monitor = False
            if self.logger:
                self.logger.warning("NVML initialization failed. GPU memory monitoring will be limited to PyTorch API.")

    def start_training(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info("Performance monitoring started.")
            
    def start_epoch(self):
        self.epoch_start_time = time.time()

    def end_epoch(self, epoch):
        if self.epoch_start_time:
            elapsed = time.time() - self.epoch_start_time
            if self.logger:
                self.logger.info(f"Epoch {epoch} duration: {elapsed:.2f} seconds")
            return elapsed
        return 0

    def record_inference_time(self, batch_size, duration):
        """Record inference time for a batch to calculate FPS"""
        if duration > 0:
            fps = batch_size / duration
            self.inference_times.append(fps)

    def get_gpu_memory_usage(self):
        """Get current and peak GPU memory usage in MB"""
        stats = {}
        
        # PyTorch memory stats
        if torch.cuda.is_available():
            current_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            current_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            stats['pytorch_allocated_mb'] = current_allocated
            stats['pytorch_peak_allocated_mb'] = peak_allocated
            stats['pytorch_reserved_mb'] = current_reserved
            
        # System NVML stats (more accurate for total device memory)
        if self.has_gpu_monitor:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                stats['total_gpu_used_mb'] = info.used / (1024 * 1024)
                stats['gpu_total_mb'] = info.total / (1024 * 1024)
            except Exception:
                pass
                
        return stats

    def get_system_memory_usage(self):
        """Get system RAM usage in MB"""
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / (1024 * 1024)
        return ram_usage

    def log_performance_summary(self):
        """Log a summary of performance metrics"""
        if not self.logger:
            return

        summary = []
        summary.append("\n" + "="*50)
        summary.append("PERFORMANCE MONITORING SUMMARY")
        summary.append("="*50)
        
        # Training Time
        if self.start_time:
            total_time = time.time() - self.start_time
            summary.append(f"Total Training Time: {total_time/3600:.2f} hours ({total_time:.2f} seconds)")

        # Inference Speed
        if self.inference_times:
            avg_fps = sum(self.inference_times) / len(self.inference_times)
            summary.append(f"Average Inference Speed: {avg_fps:.2f} images/sec")
            
        # Memory Usage
        gpu_stats = self.get_gpu_memory_usage()
        if gpu_stats:
            summary.append("-" * 30)
            summary.append("GPU Memory Usage:")
            if 'pytorch_peak_allocated_mb' in gpu_stats:
                summary.append(f"  PyTorch Peak Allocated: {gpu_stats['pytorch_peak_allocated_mb']:.2f} MB")
            if 'pytorch_allocated_mb' in gpu_stats:
                summary.append(f"  PyTorch Current Allocated: {gpu_stats['pytorch_allocated_mb']:.2f} MB")
            if 'total_gpu_used_mb' in gpu_stats:
                summary.append(f"  Total GPU Memory Used (System): {gpu_stats['total_gpu_used_mb']:.2f} MB")
        
        ram_usage = self.get_system_memory_usage()
        summary.append(f"System RAM Usage (Current Process): {ram_usage:.2f} MB")
        summary.append("="*50 + "\n")
        
        for line in summary:
            self.logger.info(line)
