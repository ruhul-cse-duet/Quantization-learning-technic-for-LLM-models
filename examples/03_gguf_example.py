"""
Example: GGUF Quantization
Demonstrates how to use GGUF format for CPU-friendly inference
"""

import sys
sys.path.append('../')

from src.quantizers import GGUFQuantizer


def main():
    print("=" * 70)
    print("GGUF Quantization Example")
    print("=" * 70)
    
    # Initialize quantizer
    # Note: Use GGUF models from TheBloke's collections
    quantizer = GGUFQuantizer(
        model_name="TheBloke/zephyr-7B-beta-GGUF",
        model_file="zephyr-7b-beta.Q4_K_M.gguf",
        model_type="mistral",
        gpu_layers=0,  # Set to 0 for CPU-only, or higher to offload layers to GPU
        context_length=2048
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
    prompt = "Explain the benefits of running LLMs on CPU: "
    
    print("\nPrompt:")
    print(prompt)
    
    # Generate response
    print("\nGenerating response...")
    # Note: GGUF uses different generation API
    output = model(
        prompt,
        max_new_tokens=150,
        temperature=0.7
    )
    
    print("\nGenerated Output:")
    print("-" * 70)
    print(output)
    print("-" * 70)
    
    # Display model info
    print(f"\n✓ Quantization: Q4_K_M (4-bit)")
    print(f"✓ Device: CPU with GPU layer offloading")
    print(f"✓ Best for: Consumer hardware without dedicated GPU")
    print(f"✓ Note: Excellent for MacBooks and laptops")
    

if __name__ == "__main__":
    main()
