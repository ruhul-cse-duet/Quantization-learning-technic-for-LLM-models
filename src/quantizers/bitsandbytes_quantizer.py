"""
BitsAndBytes Quantizer
Implements 4-bit quantization using BitsAndBytes library
"""

from typing import Tuple, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from .base_quantizer import BaseQuantizer


class BitsAndBytesQuantizer(BaseQuantizer):
    """
    BitsAndBytes 4-bit quantization implementation
    
    Uses NormalFloat4 (NF4) for efficient 4-bit quantization with:
    - Weight normalization
    - Double quantization support
    - Dynamic dequantization during inference
    
    Best for: GPU inference with minimal accuracy loss
    """
    
    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
        device: str = "auto",
        cache_dir: str = None
    ):
        """        Initialize BitsAndBytes quantizer
        
        Args:
            model_name: HuggingFace model identifier
            load_in_4bit: Whether to load model in 4-bit
            bnb_4bit_quant_type: Quantization type ('nf4' or 'fp4')
            bnb_4bit_use_double_quant: Use double quantization
            bnb_4bit_compute_dtype: Computation dtype (bfloat16 recommended)
            device: Device to load model on
            cache_dir: Cache directory for models
        """
        super().__init__(model_name, device, cache_dir)
        
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
    
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get BitsAndBytes quantization configuration"""
        return {
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": str(self.bnb_4bit_compute_dtype)
        }
    
    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model with BitsAndBytes quantization
        
        Returns:
            Tuple of (quantized_model, tokenizer)        """
        print(f"Loading {self.model_name} with BitsAndBytes 4-bit quantization...")
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Model loaded successfully!")
        print(f"  Model size: {self.get_model_size():.2f} GB")
        
        return self.model, self.tokenizer
