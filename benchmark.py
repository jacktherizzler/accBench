import torch
import time
from typing import Dict, List
import subprocess

class BenchmarkTool:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataset = None

    def set_model(self, model):
        self.model = model.to(self.device)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def run_benchmark(self, iterations: int = 10) -> Dict[str, float]:
        results = {}
        
        if self.model is None:
            return {'error': 'Model not initialized. Please set model first.'}
        if self.dataset is None:
            return {'error': 'Dataset not initialized. Please set dataset first.'}
        
        # Training benchmark
        start_time = time.time()
        for _ in range(iterations):
            self.model.train()
            for data, target in self.dataset:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
        results['training_time'] = (time.time() - start_time) / iterations

        # Inference benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                self.model.eval()
                for data, _ in self.dataset:
                    data = data.to(self.device)
                self.model(data)
        results['inference_time'] = (time.time() - start_time) / iterations

        return results

    def run_colab_benchmark(self) -> Dict[str, float]:
        try:
            import google.colab
            in_colab = True
        except:
            in_colab = False
            return {'error': 'Not running in Colab environment'}

        if in_colab:
            # Setup Colab environment
            import torch
            if not torch.cuda.is_available():
                return {'error': 'GPU not available in Colab'}

            # Run benchmark with Colab GPU
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            return self.run_benchmark()

    def run_cpu_vs_gpu_comparison(self) -> Dict[str, Dict[str, float]]:
        results = {}
        
        # CPU benchmark
        original_device = self.device
        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        results['cpu'] = self.run_benchmark()

        # GPU benchmark if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            results['gpu'] = self.run_benchmark()

        # Restore original device
        self.device = original_device
        self.model = self.model.to(self.device)

        return results

    def print_results(self, results: Dict[str, Dict[str, float]]):
        print("\n=== ML Accelerator Benchmark Results ===")
        for device, metrics in results.items():
            if 'error' in metrics:
                print(f"\n{device.upper()} Error: {metrics['error']}")
                continue
            
            print(f"\n{device.upper()} Results:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f} seconds")
            
            # Add comparison if both CPU and GPU results exist
            if 'cpu' in results and 'gpu' in results:
                speedup = results['cpu']['training_time'] / results['gpu']['training_time']
                print(f"\nGPU Speedup: {speedup:.2f}x faster than CPU")

if __name__ == "__main__":
    # Example usage
    benchmark = BenchmarkTool()
    # Set your model and dataset here
    comparison_results = benchmark.run_cpu_vs_gpu_comparison()
    benchmark.print_results(comparison_results)