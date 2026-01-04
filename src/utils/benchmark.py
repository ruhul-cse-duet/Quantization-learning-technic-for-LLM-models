"""
Benchmark Utility
Measures and compares performance of different quantization methods
"""

import time
import torch
from typing import Dict, Any, List, Optional
from .memory_tracker import MemoryTracker


class Benchmark:
    """
    Utility class for benchmarking quantized models
    """
    
    def __init__(self):
        """Initialize benchmark"""
        self.results = []
        self.memory_tracker = MemoryTracker()
    
    def benchmark_generation(
        self,
        model,
        tokenizer,
        prompts: List[str],
        max_length: int = 100,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark text generation speed
        
        Args:
            model: Model to benchmark
            tokenizer: Tokenizer for the model
            prompts: List of test prompts
            max_length: Maximum generation length
            num_runs: Number of runs for averaging
        
        Returns:
            Dictionary with benchmark results
        """
        generation_times = []
        tokens_generated = []
        
        for _ in range(num_runs):
            for prompt in prompts:
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Measure generation time
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=max_length,
                        do_sample=False
                    )
                end_time = time.time()
                
                generation_times.append(end_time - start_time)
                tokens_generated.append(outputs.shape[1] - inputs['input_ids'].shape[1])
        
        avg_time = sum(generation_times) / len(generation_times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        tokens_per_second = avg_tokens / avg_time if avg_time > 0 else 0
        
        return {
            "avg_generation_time_s": avg_time,
            "avg_tokens_generated": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "num_runs": num_runs,
            "num_prompts": len(prompts)
        }
    
    def benchmark_memory(
        self,
        load_function,
        label: str = "model"
    ) -> Dict[str, Any]:
        """
        Benchmark memory usage during model loading
        
        Args:
            load_function: Function that loads the model
            label: Label for this benchmark
        
        Returns:
            Dictionary with memory benchmark results
        """
        self.memory_tracker.start_tracking()
        
        start_time = time.time()
        result = load_function()
        load_time = time.time() - start_time
        
        self.memory_tracker.record_measurement("after_load")
        memory_report = self.memory_tracker.get_report()
        
        return {
            "label": label,
            "load_time_s": load_time,
            "memory": memory_report
        }
    
    def compare_quantizations(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare results from multiple quantization methods
        
        Args:
            results: List of benchmark results
        
        Returns:
            Comparison summary
        """
        if not results:
            return {}
        
        comparison = {
            "methods": [],
            "fastest_generation": None,
            "lowest_memory": None,
            "best_tokens_per_sec": None
        }
        
        for result in results:
            comparison["methods"].append(result.get("label", "unknown"))
        
        # Find fastest generation
        gen_times = [(r.get("label"), r.get("avg_generation_time_s", float('inf'))) 
                     for r in results if "avg_generation_time_s" in r]
        if gen_times:
            comparison["fastest_generation"] = min(gen_times, key=lambda x: x[1])
        
        # Find lowest memory
        mem_usage = [(r.get("label"), r.get("memory", {}).get("peak", {}).get("ram_gb", float('inf'))) 
                     for r in results if "memory" in r]
        if mem_usage:
            comparison["lowest_memory"] = min(mem_usage, key=lambda x: x[1])
        
        # Find best tokens/sec
        tokens_per_sec = [(r.get("label"), r.get("tokens_per_second", 0)) 
                          for r in results if "tokens_per_second" in r]
        if tokens_per_sec:
            comparison["best_tokens_per_sec"] = max(tokens_per_sec, key=lambda x: x[1])
        
        return comparison
    
    def print_report(self, result: Dict[str, Any]):
        """
        Print formatted benchmark report
        
        Args:
            result: Benchmark result dictionary
        """
        print("\n" + "="*60)
        print(f"Benchmark Report: {result.get('label', 'Unknown')}")
        print("="*60)
        
        if "load_time_s" in result:
            print(f"Load Time: {result['load_time_s']:.2f}s")
        
        if "avg_generation_time_s" in result:
            print(f"Avg Generation Time: {result['avg_generation_time_s']:.3f}s")
            print(f"Tokens/Second: {result.get('tokens_per_second', 0):.2f}")
        
        if "memory" in result:
            memory = result["memory"]
            if "peak" in memory:
                print(f"Peak RAM: {memory['peak']['ram_gb']:.2f} GB")
                if memory['peak'].get('vram_gb'):
                    print(f"Peak VRAM: {memory['peak']['vram_gb']:.2f} GB")
        
        print("="*60 + "\n")
