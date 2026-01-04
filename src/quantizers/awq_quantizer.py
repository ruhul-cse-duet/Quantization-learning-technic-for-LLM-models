"""
AWQ Quantizer  
Implements Activation-aware Weight Quantization for high-performance inference
"""

from typing import Tuple, Dict, Any, Optional
from .base_quantizer import BaseQuantizer


class AWQQuantizer(BaseQuantizer):
    """
    AWQ (Activation-aware Weight Quantization) implementation
    
    AWQ is an advanced quantization method that:
    - Analyzes activation patterns to identify important weights
    - Skips quantizing critical weights
    - Achieves better accuracy than GPTQ
    - Provides faster inference
    
    Best for: High-performance GPU deployments with best accuracy
    
    Note: Requires vLLM library for efficient inference
    """
    
    def __init__(
        self,
        model_name: str,
        bits: int = 4,
        quantization: str = "awq",
        dtype: str = "half",
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 4096,
        device: str = "cuda",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize AWQ quantizer
        
        Args:
            model_name: HuggingFace model identifier (pre-quantized AWQ model)
            bits: Number of bits for quantization            quantization: Quantization method
            dtype: Data type for model weights
            gpu_memory_utilization: GPU memory to use (0.0 to 1.0)
            max_model_len: Maximum model context length
            device: Device to use
            cache_dir: Cache directory
        """
        super().__init__(model_name, device, cache_dir)
        
        self.bits = bits
        self.quantization = quantization
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get AWQ quantization configuration"""
        return {
            "bits": self.bits,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len
        }
    
    def load_model(self) -> Tuple[Any, None]:
        """
        Load AWQ quantized model using vLLM
        
        Note: vLLM uses its own tokenizer internally
        
        Returns:
            Tuple of (vllm_model, None)
        """
        print(f"Loading {self.model_name} with AWQ quantization...")
        
        try:
            from vllm import LLM, SamplingParams        except ImportError:
            raise ImportError(
                "vLLM is required for AWQ quantization. "
                "Install it with: pip install vllm"
            )
        
        # Load model with vLLM
        self.model = LLM(
            model=self.model_name,
            quantization=self.quantization,
            dtype=self.dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len
        )
        
        # Store sampling params for generation
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=256
        )
        
        self.tokenizer = None  # vLLM handles tokenization internally
        
        print(f"✓ AWQ model loaded successfully with vLLM!")
        print(f"  Quantization: {self.bits}-bit AWQ")
        
        return self.model, self.tokenizer
    
    def generate(self, prompt: str, max_length: int = 256, **kwargs) -> str:
        """
        Generate text using vLLM
        
        Args:
            prompt: Input text
            max_length: Maximum tokens to generate
            **kwargs: Additional sampling parameters
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")        
        # Update sampling params with kwargs
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=max_length,
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.95)
        )
        
        # Generate
        outputs = self.model.generate(prompt, sampling_params)
        return outputs[0].outputs[0].text
