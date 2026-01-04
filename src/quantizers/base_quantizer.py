"""
Base Quantizer Class
Provides common interface for all quantization methods
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class BaseQuantizer(ABC):
    """
    Abstract base class for LLM quantizers
    
    All quantization methods should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the quantizer
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ('auto', 'cuda', 'cpu')
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load and quantize the model
        
        Returns:
            Tuple of (model, tokenizer)
        """
        pass
    
    @abstractmethod
    def get_quantization_config(self) -> Dict[str, Any]:
        """
        Get the quantization configuration
        
        Returns:
            Dictionary containing quantization parameters
        """
        pass
    
    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available() and self.device != "cpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_model_size(self) -> float:
        """
        Calculate model size in GB
        
        Returns:
            Model size in gigabytes
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
                
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_gb = (param_size + buffer_size) / 1024**3
        return size_gb
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
