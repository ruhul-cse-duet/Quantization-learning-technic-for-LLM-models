"""
Model Loader Utility
Handles loading and managing different quantized models
"""

import os
from typing import Optional, Dict, Any, List
import torch
from pathlib import Path


class ModelLoader:
    """
    Utility class for loading and managing quantized models
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model loader
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = cache_dir or os.path.join(Path.home(), ".cache", "huggingface")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @staticmethod
    def check_gpu_availability() -> Dict[str, Any]:
        """
        Check GPU availability and specs
        
        Returns:
            Dictionary with GPU information
        """
        gpu_info = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": []
        }
        
        if gpu_info["available"]:
            for i in range(gpu_info["device_count"]):
                device_props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "id": i,
                    "name": device_props.name,
                    "total_memory_gb": device_props.total_memory / 1024**3,
                    "compute_capability": f"{device_props.major}.{device_props.minor}"
                })
        
        return gpu_info
    
    @staticmethod
    def get_recommended_device() -> str:
        """
        Get recommended device based on availability
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def list_cached_models(self) -> List[str]:
        """
        List models cached locally
        
        Returns:
            List of cached model paths
        """
        cached_models = []
        if os.path.exists(self.cache_dir):
            for root, dirs, files in os.walk(self.cache_dir):
                if "pytorch_model.bin" in files or "model.safetensors" in files:
                    cached_models.append(root)
        return cached_models
    
    @staticmethod
    def estimate_model_size(model_name: str, precision: str = "fp32") -> float:
        """
        Estimate model size based on parameter count
        
        Args:
            model_name: Model identifier
            precision: Precision level (fp32, fp16, int8, int4)
        
        Returns:
            Estimated size in GB
        """
        # Extract parameter count from model name
        param_count = 0
        model_name_lower = model_name.lower()
        
        if "7b" in model_name_lower:
            param_count = 7
        elif "13b" in model_name_lower:
            param_count = 13
        elif "70b" in model_name_lower:
            param_count = 70
        elif "3b" in model_name_lower:
            param_count = 3
        else:
            param_count = 7  # default
        
        # Calculate size based on precision
        bytes_per_param = {
            "fp32": 4,
            "fp16": 2,
            "int8": 1,
            "int4": 0.5
        }
        
        size_gb = (param_count * 1e9 * bytes_per_param.get(precision, 4)) / 1024**3
        return size_gb
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cache directory
        
        Returns:
            Dictionary with cache statistics
        """
        cache_info = {
            "path": self.cache_dir,
            "exists": os.path.exists(self.cache_dir),
            "size_gb": 0,
            "num_files": 0
        }
        
        if cache_info["exists"]:
            total_size = 0
            num_files = 0
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                        num_files += 1
                    except:
                        pass
            
            cache_info["size_gb"] = total_size / 1024**3
            cache_info["num_files"] = num_files
        
        return cache_info
