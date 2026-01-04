"""
GPTQ Quantizer
Implements GPTQ (GPT Quantization) for efficient 4-bit GPU inference
"""

from typing import Tuple, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from .base_quantizer import BaseQuantizer


class GPTQQuantizer(BaseQuantizer):
    """
    GPTQ quantization implementation
    
    GPTQ is a post-training quantization method that:
    - Minimizes mean squared error during quantization
    - Optimizes for GPU inference
    - Provides fast inference with minimal accuracy loss
    
    Best for: Production GPU deployments
    """
    
    def __init__(
        self,
        model_name: str,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        device: str = "auto",
        cache_dir: str = None,
        revision: str = "main"
    ):
        """
        Initialize GPTQ quantizer
        
        Args:            model_name: HuggingFace model identifier (pre-quantized GPTQ model)
            bits: Number of bits for quantization
            group_size: Group size for quantization
            desc_act: Whether to use desc_act
            device: Device to load model on
            cache_dir: Cache directory for models
            revision: Model revision to use
        """
        super().__init__(model_name, device, cache_dir)
        
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.revision = revision
    
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get GPTQ quantization configuration"""
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "revision": self.revision
        }
    
    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load pre-quantized GPTQ model
        
        Note: This loads a model that has already been quantized with GPTQ.
        For quantizing a model yourself, use auto-gptq library separately.
        
        Returns:
            Tuple of (quantized_model, tokenizer)
        """
        print(f"Loading {self.model_name} with GPTQ quantization...")
                
        # Load model (pre-quantized)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=False,
            revision=self.revision,
            cache_dir=self.cache_dir
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            cache_dir=self.cache_dir
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ GPTQ model loaded successfully!")
        print(f"  Model size: {self.get_model_size():.2f} GB")
        print(f"  Quantization: {self.bits}-bit")
        
        return self.model, self.tokenizer
