"""
Example: BitsAndBytes 4-bit Quantization
Demonstrates how to use BitsAndBytes for efficient GPU inference
"""

import sys
sys.path.append('../llm-quantization-learning')

from src.quantizers import BitsAndBytesQuantizer


def main():
    print("=" * 70)
    print("BitsAndBytes 4-bit Quantization Example")
    print("=" * 70)
    
    # Initialize quantizer
    quantizer = BitsAndBytesQuantizer(
        model_name="HuggingFaceH4/zephyr-7b-beta",
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
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
You are a friendly chatbot.</s>
<|user|>
Explain quantization in simple terms.</s>
<|assistant|>
"""
    
    print("\nPrompt:")
    print(prompt)
    
    # Generate response
    print("\nGenerating response...")
    output = quantizer.generate(        prompt=prompt,
        max_length=200,
        temperature=0.7
    )
    
    print("\nGenerated Output:")
    print("-" * 70)
    print(output)
    print("-" * 70)
    
    # Display model info
    print(f"\n✓ Model Size: {quantizer.get_model_size():.2f} GB")
    print(f"✓ Quantization: 4-bit NormalFloat (NF4)")
    print(f"✓ Device: GPU (auto)")
    

if __name__ == "__main__":
    main()
