"""
Example: AWQ (Activation-aware Weight Quantization)
Demonstrates how to use AWQ for high-performance GPU inference
"""

import sys
sys.path.append('../')

from src.quantizers import AWQQuantizer


def main():
    print("=" * 70)
    print("AWQ 4-bit Quantization Example")
    print("=" * 70)
    
    # Initialize quantizer
    # Note: Use pre-quantized AWQ models
    quantizer = AWQQuantizer(
        model_name="TheBloke/zephyr-7B-beta-AWQ",
        bits=4,
        quantization="awq",
        dtype="half",
        gpu_memory_utilization=0.90,
        max_model_len=4096
    )
    
    # Display configuration
    print("\nQuantization Configuration:")
    config = quantizer.get_quantization_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load model
    print("\nLoading model with vLLM...")
    model, _ = quantizer.load_model()
    
    # Prepare prompt
    prompt = """<|system|>
You are a knowledgeable AI assistant.</s>
<|user|>
Compare AWQ to other quantization methods.</s>
<|assistant|>
"""
    
    print("\nPrompt:")
    print(prompt)
    
    # Generate response
    print("\nGenerating response...")
    output = quantizer.generate(
        prompt=prompt,
        max_length=200,
        temperature=0.7
    )
    
    print("\nGenerated Output:")
    print("-" * 70)
    print(output)
    print("-" * 70)
    
    # Display model info
    print(f"\n✓ Quantization: 4-bit AWQ")
    print(f"✓ Device: GPU with vLLM")
    print(f"✓ Best for: High-performance GPU deployments")
    print(f"✓ Advantages: Better accuracy than GPTQ, faster inference")
    

if __name__ == "__main__":
    main()
