# LLM Quantization Learning Project

A comprehensive, modular project for learning and implementing various Large Language Model (LLM) quantization techniques.

## 📚 Overview

This project provides hands-on implementations of different quantization methods to reduce LLM memory footprint and improve inference efficiency while maintaining model performance.

## 🎯 Quantization Techniques Covered

### 1. **BitsAndBytes (4-bit NF4)**
- **Type**: Post-training quantization
- **Target**: GPU inference
- **Precision**: 4-bit (NormalFloat4)
- **Use Case**: Efficient GPU deployment with minimal accuracy loss
- **Key Features**:
  - Normalized Float 4-bit quantization
  - Double quantization support
  - Dynamic dequantization during inference

### 2. **GPTQ (GPT Quantization)**
- **Type**: Post-training quantization
- **Target**: GPU inference
- **Precision**: 4-bit
- **Use Case**: Production GPU deployments
- **Key Features**:
  - Weight-only quantization
  - Minimizes mean squared error
  - Fast inference on GPU

### 3. **GGUF (GPT-Generated Unified Format)**
- **Type**: Quantized model format
- **Target**: CPU & GPU hybrid inference
- **Precision**: Multiple (2-bit to 8-bit)- **Use Case**: Running LLMs on consumer hardware (CPU or limited GPU)
- **Key Features**:
  - CPU-friendly format
  - Layer offloading to GPU
  - Excellent for MacBooks and consumer PCs

### 4. **AWQ (Activation-aware Weight Quantization)**
- **Type**: Activation-aware quantization
- **Target**: GPU inference
- **Precision**: 4-bit
- **Use Case**: High-performance GPU deployment
- **Key Features**:
  - Skips critical weights
  - Better accuracy preservation
  - Faster than GPTQ

## 📁 Project Structure

```
llm-quantization-learning/
├── src/
│   ├── quantizers/
│   │   ├── __init__.py
│   │   ├── bitsandbytes_quantizer.py
│   │   ├── gptq_quantizer.py
│   │   ├── gguf_quantizer.py
│   │   └── awq_quantizer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   ├── memory_tracker.py
│   │   └── benchmark.py│   └── main.py
├── examples/
│   ├── 01_bitsandbytes_example.py
│   ├── 02_gptq_example.py
│   ├── 03_gguf_example.py
│   ├── 04_awq_example.py
│   └── 05_comparison.py
├── config/
│   └── model_config.yaml
├── docs/
│   ├── quantization_theory.md
│   └── performance_comparison.md
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-quantization-learning.git
cd llm-quantization-learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.quantizers import BitsAndBytesQuantizer

# Initialize quantizer
quantizer = BitsAndBytesQuantizer(    model_name="HuggingFaceH4/zephyr-7b-beta",
    load_in_4bit=True
)

# Load and quantize model
model, tokenizer = quantizer.load_model()

# Generate text
prompt = "What is quantization?"
output = quantizer.generate(prompt, max_length=100)
print(output)
```

## 📊 Comparison Table

| Technique | Precision | Target Hardware | Speed | Memory | Accuracy | Best For |
|-----------|-----------|-----------------|-------|---------|----------|----------|
| BitsAndBytes | 4-bit | GPU | Fast | Low | Good | GPU deployments |
| GPTQ | 4-bit | GPU | Very Fast | Low | Good | Production GPU |
| GGUF | 2-8 bit | CPU/GPU | Medium | Very Low | Variable | Consumer hardware |
| AWQ | 4-bit | GPU | Fastest | Low | Best | High-performance GPU |

## 🔧 Configuration

Edit `config/model_config.yaml` to customize:

```yaml
model:
  name: "HuggingFaceH4/zephyr-7b-beta"
  cache_dir: "./models"

quantization:
  bitsandbytes:
    load_in_4bit: true
    bnb_4bit_quant_type: "nf4"
    bnb_4bit_use_double_quant: true    
  gptq:
    bits: 4
    group_size: 128
    
  gguf:
    model_type: "mistral"
    gpu_layers: 50
    
  awq:
    bits: 4
```

## 📖 Learning Resources

### Theory Documentation
- [Quantization Theory](quantization_theory.md)
- [Performance Comparison](docs/performance_comparison.md)

### Examples
1. **BitsAndBytes**: Basic 4-bit quantization with NF4
2. **GPTQ**: GPU-optimized weight quantization
3. **GGUF**: CPU-friendly format with layer offloading
4. **AWQ**: Activation-aware quantization
5. **Comparison**: Side-by-side comparison of all methods

## 🎓 Key Concepts

### What is Quantization?
Quantization reduces the precision of model weights and activations from 32-bit floating point to lower bit representations (8-bit, 4-bit, or even 2-bit), dramatically reducing:
- Model size (2-8x reduction)
- Memory requirements
- Inference latency

### Trade-offs
- **Precision**: Lower bits = smaller models but potential accuracy loss
- **Hardware**: Some methods optimized for GPU, others for CPU
- **Speed**: Quantization can improve inference speed significantly

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- accelerate
- bitsandbytes
- auto-gptq
- ctransformers
- vllm (for AWQ)
- pyyaml

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - feel free to use for learning and projects

## 🙏 Acknowledgments

- HuggingFace for transformers library
- TheBloke for pre-quantized models
- Research papers on quantization techniques

## 📧 Contact

Ruhul Amin

LindeIn: (https://www.linkedin.com/in/ruhul-duet-cse/)

For questions or support, please open an issue or
contact: ruhul.cse.duet@gmail.com



---

**Happy Learning! 🚀**
