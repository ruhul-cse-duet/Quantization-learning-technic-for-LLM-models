"""
Memory Tracker Utility
Monitors memory usage during model loading and inference
"""

import psutil
import torch
from typing import Dict, Any, Optional
import time


class MemoryTracker:
    """
    Utility class for tracking memory usage
    """
    
    def __init__(self):
        """Initialize memory tracker"""
        self.start_ram = None
        self.start_vram = None
        self.measurements = []
    
    def start_tracking(self):
        """Start tracking memory usage"""
        self.start_ram = self.get_ram_usage()
        self.start_vram = self.get_vram_usage()
        self.measurements = []
        self.record_measurement("start")
    
    def record_measurement(self, label: str = "checkpoint"):
        """
        Record current memory state
        
        Args:
            label: Label for this measurement
        """
        measurement = {
            "label": label,
            "timestamp": time.time(),
            "ram_gb": self.get_ram_usage(),
            "vram_gb": self.get_vram_usage()
        }
        self.measurements.append(measurement)
    
    @staticmethod
    def get_ram_usage() -> float:
        """
        Get current RAM usage
        
        Returns:
            RAM usage in GB
        """
        process = psutil.Process()
        return process.memory_info().rss / 1024**3
    
    @staticmethod
    def get_vram_usage() -> Optional[float]:
        """
        Get current VRAM usage
        
        Returns:
            VRAM usage in GB (None if no GPU)
        """
        if not torch.cuda.is_available():
            return None
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return max(allocated, reserved)
    
    def get_memory_delta(self) -> Dict[str, Any]:
        """
        Get memory usage change since tracking started
        
        Returns:
            Dictionary with memory deltas
        """
        if not self.start_ram:
            raise ValueError("Call start_tracking() first")
        
        current_ram = self.get_ram_usage()
        current_vram = self.get_vram_usage()
        
        return {
            "ram_delta_gb": current_ram - self.start_ram,
            "vram_delta_gb": (current_vram - self.start_vram) if current_vram and self.start_vram else None,
            "current_ram_gb": current_ram,
            "current_vram_gb": current_vram
        }
    
    def get_peak_memory(self) -> Dict[str, float]:
        """
        Get peak memory usage from measurements
        
        Returns:
            Dictionary with peak RAM and VRAM
        """
        if not self.measurements:
            return {"ram_gb": 0, "vram_gb": 0}
        
        peak_ram = max(m["ram_gb"] for m in self.measurements)
        vram_values = [m["vram_gb"] for m in self.measurements if m["vram_gb"] is not None]
        peak_vram = max(vram_values) if vram_values else None
        
        return {
            "ram_gb": peak_ram,
            "vram_gb": peak_vram
        }
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate memory usage report
        
        Returns:
            Comprehensive memory report
        """
        return {
            "measurements": self.measurements,
            "delta": self.get_memory_delta(),
            "peak": self.get_peak_memory()
        }
