"""
Benchmarking tools for deepfake detection models.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import GPUtil

from .metrics import DetectionMetrics, BenchmarkSuite, ModelAnalyzer
from ..models import create_cnn_model, create_transformer_model, create_ensemble_model

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    batch_sizes: List[int] = None
    input_sizes: List[Tuple[int, int]] = None
    num_warmup_runs: int = 5
    num_benchmark_runs: int = 20
    device: str = "auto"
    precision: str = "fp32"  # fp32, fp16, int8
    memory_analysis: bool = True
    energy_analysis: bool = False
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]
        if self.input_sizes is None:
            self.input_sizes = [(224, 224), (256, 256), (384, 384)]

class ModelBenchmarker:
    """Comprehensive model benchmarking tool."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results = {}
        self.device = self._setup_device()
        
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def benchmark_single_model(self, 
                              model: nn.Module,
                              model_name: str,
                              input_shape: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Benchmark a single model configuration."""
        model = model.to(self.device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(self.config.num_warmup_runs):
                _ = model(dummy_input)
        
        # Synchronize if using GPU
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark runs
        times = []
        memory_usage = []
        
        for run in range(self.config.num_benchmark_runs):
            # Memory tracking
            if self.config.memory_analysis and self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Memory usage
            if self.config.memory_analysis and self.device.type == "cuda":
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
        
        # Compute statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        throughput = input_shape[0] / avg_time  # samples per second
        
        result = {
            'model_name': model_name,
            'batch_size': input_shape[0],
            'input_size': input_shape[2:],
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'throughput_fps': throughput,
            'device': str(self.device)
        }
        
        if memory_usage:
            result.update({
                'avg_memory_usage_mb': np.mean(memory_usage),
                'max_memory_usage_mb': np.max(memory_usage),
                'std_memory_usage_mb': np.std(memory_usage)
            })
        
        return result
    
    def benchmark_model_variants(self, 
                                model_configs: List[Dict],
                                batch_sizes: List[int] = None,
                                input_sizes: List[Tuple[int, int]] = None) -> Dict[str, List[Dict]]:
        """Benchmark multiple model variants."""
        batch_sizes = batch_sizes or self.config.batch_sizes
        input_sizes = input_sizes or self.config.input_sizes
        
        all_results = {}
        
        for config in model_configs:
            model_name = config['name']
            model_type = config['type']
            model_params = config.get('params', {})
            
            model_results = []
            
            for batch_size in batch_sizes:
                for input_size in input_sizes:
                    # Create model
                    if model_type == 'cnn':
                        model = create_cnn_model(**model_params)
                    elif model_type == 'transformer':
                        model = create_transformer_model(**model_params)
                    elif model_type == 'ensemble':
                        model = create_ensemble_model(**model_params)
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")
                    
                    # Benchmark
                    input_shape = (batch_size, 3, input_size[0], input_size[1])
                    result = self.benchmark_single_model(model, model_name, input_shape)
                    model_results.append(result)
            
            all_results[model_name] = model_results
        
        return all_results
    
    def compare_models(self, results: Dict[str, List[Dict]]) -> pd.DataFrame:
        """Compare model performance across different configurations."""
        all_data = []
        
        for model_name, model_results in results.items():
            for result in model_results:
                result['model_name'] = model_name
                all_data.append(result)
        
        df = pd.DataFrame(all_data)
        return df
    
    def generate_performance_report(self, 
                                   results: Dict[str, List[Dict]],
                                   output_dir: str = "benchmark_results") -> str:
        """Generate comprehensive performance report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Convert results to DataFrame
        df = self.compare_models(results)
        
        # Save raw results
        df.to_csv(output_path / "benchmark_results.csv", index=False)
        
        # Generate plots
        self._create_performance_plots(df, output_path)
        
        # Generate summary report
        summary = self._generate_summary_report(df)
        
        with open(output_path / "summary_report.txt", "w") as f:
            f.write(summary)
        
        return str(output_path / "summary_report.txt")
    
    def _create_performance_plots(self, df: pd.DataFrame, output_path: Path):
        """Create performance visualization plots."""
        plt.style.use('seaborn-v0_8')
        
        # Throughput vs Batch Size
        plt.figure(figsize=(12, 8))
        for model_name in df['model_name'].unique():
            model_data = df[df['model_name'] == model_name]
            plt.plot(model_data['batch_size'], model_data['throughput_fps'], 
                    marker='o', label=model_name, linewidth=2)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (FPS)')
        plt.title('Model Throughput vs Batch Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / "throughput_vs_batch_size.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Inference Time vs Batch Size
        plt.figure(figsize=(12, 8))
        for model_name in df['model_name'].unique():
            model_data = df[df['model_name'] == model_name]
            plt.plot(model_data['batch_size'], model_data['avg_inference_time'], 
                    marker='s', label=model_name, linewidth=2)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Average Inference Time (s)')
        plt.title('Inference Time vs Batch Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / "inference_time_vs_batch_size.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Memory Usage vs Batch Size
        if 'avg_memory_usage_mb' in df.columns:
            plt.figure(figsize=(12, 8))
            for model_name in df['model_name'].unique():
                model_data = df[df['model_name'] == model_name]
                plt.plot(model_data['batch_size'], model_data['avg_memory_usage_mb'], 
                        marker='^', label=model_name, linewidth=2)
            
            plt.xlabel('Batch Size')
            plt.ylabel('Average Memory Usage (MB)')
            plt.title('Memory Usage vs Batch Size')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path / "memory_usage_vs_batch_size.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Performance comparison heatmap
        plt.figure(figsize=(14, 10))
        pivot_table = df.pivot_table(values='throughput_fps', 
                                   index='model_name', 
                                   columns='batch_size', 
                                   aggfunc='mean')
        
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Throughput Heatmap (FPS)')
        plt.xlabel('Batch Size')
        plt.ylabel('Model Name')
        plt.savefig(output_path / "throughput_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, df: pd.DataFrame) -> str:
        """Generate text summary report."""
        report = []
        report.append("=" * 60)
        report.append("DEEPFAKE DETECTION MODEL BENCHMARK REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS:")
        report.append("-" * 20)
        report.append(f"Total models tested: {df['model_name'].nunique()}")
        report.append(f"Total configurations: {len(df)}")
        report.append(f"Device: {df['device'].iloc[0]}")
        report.append("")
        
        # Best performing models
        report.append("BEST PERFORMING MODELS:")
        report.append("-" * 25)
        
        # Highest throughput
        max_throughput = df.loc[df['throughput_fps'].idxmax()]
        report.append(f"Highest Throughput: {max_throughput['model_name']} "
                     f"({max_throughput['throughput_fps']:.1f} FPS)")
        
        # Lowest inference time
        min_time = df.loc[df['avg_inference_time'].idxmin()]
        report.append(f"Lowest Inference Time: {min_time['model_name']} "
                     f"({min_time['avg_inference_time']*1000:.1f} ms)")
        
        # Most memory efficient
        if 'avg_memory_usage_mb' in df.columns:
            min_memory = df.loc[df['avg_memory_usage_mb'].idxmin()]
            report.append(f"Most Memory Efficient: {min_memory['model_name']} "
                         f"({min_memory['avg_memory_usage_mb']:.1f} MB)")
        
        report.append("")
        
        # Model-specific summaries
        report.append("MODEL-SPECIFIC SUMMARIES:")
        report.append("-" * 25)
        
        for model_name in df['model_name'].unique():
            model_data = df[df['model_name'] == model_name]
            report.append(f"\n{model_name}:")
            report.append(f"  Average Throughput: {model_data['throughput_fps'].mean():.1f} FPS")
            report.append(f"  Average Inference Time: {model_data['avg_inference_time'].mean()*1000:.1f} ms")
            report.append(f"  Best Batch Size: {model_data.loc[model_data['throughput_fps'].idxmax(), 'batch_size']}")
            
            if 'avg_memory_usage_mb' in df.columns:
                report.append(f"  Average Memory Usage: {model_data['avg_memory_usage_mb'].mean():.1f} MB")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

class RealTimeBenchmarker:
    """Benchmarking specifically for real-time applications."""
    
    def __init__(self, target_fps: float = 30.0, tolerance: float = 0.1):
        self.target_fps = target_fps
        self.tolerance = tolerance
        self.max_frame_time = 1.0 / target_fps
    
    def benchmark_realtime_performance(self, 
                                      model: nn.Module,
                                      input_shape: Tuple[int, int, int, int],
                                      device: torch.device,
                                      num_frames: int = 1000) -> Dict[str, float]:
        """Benchmark real-time performance characteristics."""
        model = model.to(device)
        model.eval()
        
        frame_times = []
        detection_times = []
        dropped_frames = 0
        
        with torch.no_grad():
            for i in range(num_frames):
                # Simulate frame arrival
                frame_start = time.time()
                
                # Create dummy input
                dummy_input = torch.randn(input_shape).to(device)
                
                # Detection
                detection_start = time.time()
                _ = model(dummy_input)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                
                detection_end = time.time()
                frame_end = time.time()
                
                # Record times
                frame_time = frame_end - frame_start
                detection_time = detection_end - detection_start
                
                frame_times.append(frame_time)
                detection_times.append(detection_time)
                
                # Check for dropped frames
                if frame_time > self.max_frame_time:
                    dropped_frames += 1
        
        # Compute statistics
        avg_frame_time = np.mean(frame_times)
        avg_detection_time = np.mean(detection_times)
        std_frame_time = np.std(frame_times)
        
        actual_fps = 1.0 / avg_frame_time
        dropped_frame_rate = dropped_frames / num_frames
        
        # Real-time suitability
        is_realtime_suitable = (avg_frame_time <= self.max_frame_time and 
                              dropped_frame_rate <= self.tolerance)
        
        return {
            'avg_frame_time': avg_frame_time,
            'avg_detection_time': avg_detection_time,
            'std_frame_time': std_frame_time,
            'actual_fps': actual_fps,
            'target_fps': self.target_fps,
            'dropped_frames': dropped_frames,
            'dropped_frame_rate': dropped_frame_rate,
            'is_realtime_suitable': is_realtime_suitable,
            'fps_efficiency': actual_fps / self.target_fps
        }

class ScalabilityAnalyzer:
    """Analyze model scalability across different hardware configurations."""
    
    def __init__(self):
        self.hardware_info = self._get_hardware_info()
    
    def _get_hardware_info(self) -> Dict[str, any]:
        """Get current hardware information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available()
        }
        
        if info['gpu_available']:
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(info['gpu_count'])]
            
            # GPU memory
            gpus = GPUtil.getGPUs()
            if gpus:
                info['gpu_memory_gb'] = [gpu.memoryTotal / 1024 for gpu in gpus]
        
        return info
    
    def analyze_scalability(self, 
                          model: nn.Module,
                          base_input_shape: Tuple[int, int, int, int],
                          scale_factors: List[float] = None) -> Dict[str, List[Dict]]:
        """Analyze model performance across different scales."""
        scale_factors = scale_factors or [0.5, 1.0, 1.5, 2.0, 4.0]
        
        results = {
            'batch_size_scaling': [],
            'input_size_scaling': [],
            'model_size_scaling': []
        }
        
        # Batch size scaling
        for scale in scale_factors:
            scaled_batch_size = max(1, int(base_input_shape[0] * scale))
            scaled_input_shape = (scaled_batch_size, *base_input_shape[1:])
            
            benchmarker = ModelBenchmarker()
            result = benchmarker.benchmark_single_model(model, "scaled_model", scaled_input_shape)
            result['scale_factor'] = scale
            results['batch_size_scaling'].append(result)
        
        # Input size scaling
        for scale in scale_factors:
            scaled_height = int(base_input_shape[2] * scale)
            scaled_width = int(base_input_shape[3] * scale)
            scaled_input_shape = (*base_input_shape[:2], scaled_height, scaled_width)
            
            benchmarker = ModelBenchmarker()
            result = benchmarker.benchmark_single_model(model, "scaled_model", scaled_input_shape)
            result['scale_factor'] = scale
            results['input_size_scaling'].append(result)
        
        return results

def run_comprehensive_benchmark(model_configs: List[Dict],
                               output_dir: str = "benchmark_results",
                               config: BenchmarkConfig = None) -> str:
    """Run comprehensive benchmark suite."""
    config = config or BenchmarkConfig()
    benchmarker = ModelBenchmarker(config)
    
    # Run benchmarks
    results = benchmarker.benchmark_model_variants(model_configs)
    
    # Generate report
    report_path = benchmarker.generate_performance_report(results, output_dir)
    
    return report_path

# Example model configurations for benchmarking
EXAMPLE_MODEL_CONFIGS = [
    {
        'name': 'ResNet50',
        'type': 'cnn',
        'params': {'model_type': 'resnet50', 'pretrained': True}
    },
    {
        'name': 'EfficientNet-B0',
        'type': 'cnn', 
        'params': {'model_type': 'efficientnet_b0', 'pretrained': True}
    },
    {
        'name': 'ViT-Base',
        'type': 'transformer',
        'params': {'model_type': 'vit_base', 'pretrained': True}
    },
    {
        'name': 'Multi-Scale CNN',
        'type': 'cnn',
        'params': {'model_type': 'multiscale'}
    },
    {
        'name': 'Simple Ensemble',
        'type': 'ensemble',
        'params': {'ensemble_type': 'weighted_average'}
    }
]
