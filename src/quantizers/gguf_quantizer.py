"""
GGUF Quantizer
Implements GGUF format loading for CPU-friendly inference
"""

from typing import Tuple, Dict, Any
from ctransformers import AutoModelForCausalLM as CTAutoModel
from transformers import AutoTokenizer, PreTrainedTokenizer
from .base_quantizer import BaseQuantizer


class GGUFQuantizer(BaseQuantizer):
    """
    GGUF (GPT-Generated Unified Format) implementation
    
    GGUF is a quantized model format that:
    - Enables CPU inference
    - Supports GPU layer offloading
    - Works well on consumer hardware (MacBooks, PCs)
    - Multiple quantization levels (Q2_K to Q8_0)
    
    Best for: Running LLMs on consumer hardware without dedicated GPU
    """
    
    def __init__(
        self,
        model_name: str,
        model_file: str,
        model_type: str = "mistral",
        gpu_layers: int = 0,
        context_length: int = 2048,
        device: str = "cpu",
        cache_dir: str = None
    ):
        """
        Initialize GGUF quantizer
        
        Args:
            model_name: HuggingFace model identifier
            model_file: Specific GGUF file to load (e.g., 'model.Q4_K_M.gguf')            model_type: Model architecture type
            gpu_layers: Number of layers to offload to GPU
            context_length: Context window size
            device: Device preference
            cache_dir: Cache directory for models
        """
        super().__init__(model_name, device, cache_dir)
        
        self.model_file = model_file
        self.model_type = model_type
        self.gpu_layers = gpu_layers
        self.context_length = context_length
    
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get GGUF quantization configuration"""
        return {
            "model_file": self.model_file,
            "model_type": self.model_type,
            "gpu_layers": self.gpu_layers,
            "context_length": self.context_length
        }
    
    def load_model(self) -> Tuple[Any, PreTrainedTokenizer]:
        """
        Load GGUF quantized model
        
        Returns:
            Tuple of (gguf_model, tokenizer)
        """
        print(f"Loading {self.model_name} with GGUF quantization...")
        print(f"  Model file: {self.model_file}")
        print(f"  GPU layers: {self.gpu_layers}")
        
        # Load GGUF model using ctransformers
        self.model = CTAutoModel.from_pretrained(
            self.model_name,            model_file=self.model_file,
            model_type=self.model_type,
            gpu_layers=self.gpu_layers,
            context_length=self.context_length,
            hf=True
        )
        
        # Load tokenizer from original model
        # Extract base model name for tokenizer
        base_model = self.model_name.replace("-GGUF", "")
        if "TheBloke" in base_model:
            # Map TheBloke models to original models
            base_model = base_model.replace("TheBloke/", "")
            base_model = base_model.rsplit("-GGUF", 1)[0]
            base_model = "HuggingFaceH4/" + base_model.replace("-", "-").lower()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            cache_dir=self.cache_dir
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ GGUF model loaded successfully!")
        print(f"  Quantization: {self.model_file.split('.')[-2]}")
        
        return self.model, self.tokenizer
