"""
Example: Quantization Methods Comparison
Compare different quantization techniques side-by-side
"""

import sys
sys.path.append('../llm-quantization-learning')

from src.quantizers import (
    BitsAndBytesQuantizer,
    GPTQQuantizer,
    GGUFQuantizer
)
import time


def test_quantizer(quantizer, prompt, name):
    """Test a quantizer and return metrics"""
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print('=' * 70)
    
    # Load model
    start_time = time.time()
    model, tokenizer = quantizer.load_model()
    load_time = time.time() - start_time
    
    # Get model size
    try:
        model_size = quantizer.get_model_size()
    except:
        model_size = 0  # GGUF models don't support this method
    
    # Generate
    start_time = time.time()
    output = quantizer.generate(prompt, max_length=100)
    gen_time = time.time() - start_time
    
    return {
        'name': name,
        'load_time': load_time,
        'model_size': model_size,
        'gen_time': gen_time,
        'output': output
    }


def main():
    prompt = """<|system|>
You are a helpful assistant.</s>
<|user|>
What are the benefits of quantization?</s><|assistant|>
"""
    
    results = []
    
    # Test BitsAndBytes
    print("\n🔧 Testing BitsAndBytes Quantization...")
    try:
        bnb = BitsAndBytesQuantizer(
            model_name="HuggingFaceH4/zephyr-7b-beta",
            load_in_4bit=True
        )
        results.append(test_quantizer(bnb, prompt, "BitsAndBytes (4-bit NF4)"))
    except Exception as e:
        print(f"❌ BitsAndBytes failed: {e}")
    
    # Test GPTQ
    print("\n🔧 Testing GPTQ Quantization...")
    try:
        gptq = GPTQQuantizer(
            model_name="TheBloke/zephyr-7B-beta-GPTQ",
            bits=4
        )
        results.append(test_quantizer(gptq, prompt, "GPTQ (4-bit)"))
    except Exception as e:
        print(f"❌ GPTQ failed: {e}")
    
    # Test GGUF
    print("\n🔧 Testing GGUF Quantization...")
    try:
        gguf = GGUFQuantizer(
            model_name="TheBloke/zephyr-7B-beta-GGUF",
            model_file="zephyr-7b-beta.Q4_K_M.gguf",
            model_type="mistral",
            gpu_layers=20
        )
        results.append(test_quantizer(gguf, prompt, "GGUF (Q4_K_M)"))
    except Exception as e:
        print(f"❌ GGUF failed: {e}")
        
    # Display Results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n{'Method':<25} {'Load Time':<12} {'Model Size':<12} {'Gen Time'}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['name']:<25} {result['load_time']:>10.2f}s "
              f"{result['model_size']:>10.2f}GB {result['gen_time']:>8.2f}s")
    
    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result['name']}:")
        print("-" * 70)
        print(result['output'][:200] + "...")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✓ BitsAndBytes: Best for GPU with easy setup")
    print("✓ GPTQ: Fastest GPU inference, production-ready")
    print("✓ GGUF: Best for CPU/hybrid, consumer hardware")
    print("✓ AWQ: Best accuracy + speed (requires vLLM)")


if __name__ == "__main__":
    main()
