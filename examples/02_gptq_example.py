"""
Example: GPTQ 4-bit Quantization
Demonstrates how to use GPTQ for production GPU inference
"""

import sys
sys.path.append('../')

from src.quantizers import GPTQQuantizer


def main():
    print("=" * 70)
    print("GPTQ 4-bit Quantization Example")
    print("=" * 70)
    
    # Initialize quantizer with pre-quantized GPTQ model
    # Note: Use models from TheBloke's GPTQ collections
    quantizer = GPTQQuantizer(
        model_name="TheBloke/zephyr-7B-beta-GPTQ",
        bits=4,
        group_size=128,
        desc_act=False
    )
    
    # Display configuration
    print("\nQuantization Configuration:")
    config = quantizer.get_quantization_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = quantizer.load_model()
    
    # Prepare prompt
    prompt = """<|system|>
You are a helpful AI assistant.</s>
<|user|>
What are the benefits of model quantization?</s>
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
    print(f"\n✓ Model Size: {quantizer.get_model_size():.2f} GB")
    print(f"✓ Quantization: 4-bit GPTQ")
    print(f"✓ Device: GPU")
    print(f"✓ Best for: Production GPU deployments")
    

if __name__ == "__main__":
    main()
