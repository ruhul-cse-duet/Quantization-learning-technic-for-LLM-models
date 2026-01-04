"""
Main Entry Point for LLM Quantization Project
Provides CLI interface for running different quantization methods
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

from quantizers import (
    BitsAndBytesQuantizer,
    GPTQQuantizer,
    GGUFQuantizer,
    AWQQuantizer
)
from utils import ModelLoader, MemoryTracker, Benchmark


def load_config(config_path: str = "config/model_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_bitsandbytes(config: Dict[str, Any]):
    """Run BitsAndBytes quantization"""
    print("\n" + "="*70)
    print("Running BitsAndBytes 4-bit Quantization")
    print("="*70)
    
    bnb_config = config['quantization']['bitsandbytes']
    
    quantizer = BitsAndBytesQuantizer(
        model_name=config['model']['name'],
        load_in_4bit=bnb_config['load_in_4bit'],
        bnb_4bit_quant_type=bnb_config['bnb_4bit_quant_type'],
        bnb_4bit_use_double_quant=bnb_config['bnb_4bit_use_double_quant'],
        cache_dir=config['model']['cache_dir']
    )
    
    model, tokenizer = quantizer.load_model()
    
    # Test generation
    prompt = "Explain what quantization is in machine learning:"
    output = quantizer.generate(prompt, max_length=150)
    print(f"\nGenerated Text:\n{output}\n")
    
    return quantizer


def run_gptq(config: Dict[str, Any]):
    """Run GPTQ quantization"""
    print("\n" + "="*70)
    print("Running GPTQ Quantization")
    print("="*70)
    
    gptq_config = config['quantization']['gptq']
    
    # Note: Using pre-quantized GPTQ model
    model_name = config['model']['gptq_model_name']
    
    quantizer = GPTQQuantizer(
        model_name=model_name,
        bits=gptq_config['bits'],
        group_size=gptq_config['group_size'],
        cache_dir=config['model']['cache_dir']
    )
    
    model, tokenizer = quantizer.load_model()
    
    prompt = "Explain what quantization is in machine learning:"
    output = quantizer.generate(prompt, max_length=150)
    print(f"\nGenerated Text:\n{output}\n")
    
    return quantizer


def run_comparison(config: Dict[str, Any]):
    """Run comparison of all quantization methods"""
    print("\n" + "="*70)
    print("Comparing Quantization Methods")
    print("="*70)
    
    benchmark = Benchmark()
    results = []
    
    # Check GPU availability
    loader = ModelLoader()
    gpu_info = loader.check_gpu_availability()
    print(f"\nGPU Available: {gpu_info['available']}")
    
    if gpu_info['available']:
        # BitsAndBytes
        print("\n--- Testing BitsAndBytes ---")
        quantizer_bnb = run_bitsandbytes(config)
        benchmark.print_report({"label": "BitsAndBytes", "load_time_s": 0})
    
    print("\nComparison complete!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LLM Quantization Tool")
    parser.add_argument(
        '--method',
        type=str,
        choices=['bitsandbytes', 'gptq', 'gguf', 'awq', 'compare', 'all'],
        default='bitsandbytes',
        help='Quantization method to use'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/model_config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run selected method
    if args.method == 'bitsandbytes':
        run_bitsandbytes(config)
    elif args.method == 'gptq':
        run_gptq(config)
    elif args.method == 'compare' or args.method == 'all':
        run_comparison(config)
    else:
        print(f"Method {args.method} not yet implemented")


if __name__ == "__main__":
    main()
