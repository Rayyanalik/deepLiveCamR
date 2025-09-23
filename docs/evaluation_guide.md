# Evaluation Guide

This guide provides comprehensive instructions for evaluating deepfake detection models using the research platform.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Evaluation Metrics](#evaluation-metrics)
3. [Benchmarking](#benchmarking)
4. [Performance Analysis](#performance-analysis)
5. [Real-time Evaluation](#real-time-evaluation)
6. [Best Practices](#best-practices)

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- Required dependencies (see requirements.txt)

### Basic Evaluation Setup

```python
from evaluation import DetectionMetrics, BenchmarkSuite
from models import create_cnn_model
from data import create_dataset_from_directory

# Load model
model = create_cnn_model(model_name="resnet50", pretrained=True)

# Load dataset
dataset = create_dataset_from_directory("data/synthetic")

# Initialize metrics
metrics = DetectionMetrics()
```

## Evaluation Metrics

### Classification Metrics

#### Accuracy
Measures overall correctness of predictions.

```python
# Calculate accuracy
accuracy = metrics.compute_metrics()['accuracy']
print(f"Accuracy: {accuracy:.3f}")
```

#### Precision and Recall
- **Precision**: True positive rate
- **Recall**: Sensitivity to fake detection

```python
metrics = metrics.compute_metrics()
precision = metrics['precision']
recall = metrics['recall']
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
```

#### F1-Score
Harmonic mean of precision and recall.

```python
f1_score = metrics['f1_score']
print(f"F1-Score: {f1_score:.3f}")
```

#### ROC-AUC
Area under the Receiver Operating Characteristic curve.

```python
roc_auc = metrics['roc_auc']
print(f"ROC-AUC: {roc_auc:.3f}")
```

### Performance Metrics

#### Inference Time
Time required for single prediction.

```python
avg_inference_time = metrics['avg_inference_time_ms']
print(f"Average Inference Time: {avg_inference_time:.1f} ms")
```

#### Throughput
Predictions per second.

```python
throughput = metrics['throughput_fps']
print(f"Throughput: {throughput:.1f} FPS")
```

#### Memory Usage
RAM consumption during inference.

```python
memory_usage = metrics['avg_memory_usage_mb']
print(f"Memory Usage: {memory_usage:.1f} MB")
```

## Benchmarking

### Model Benchmarking

```python
from evaluation import ModelBenchmarker, BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    batch_sizes=[1, 4, 8, 16, 32],
    input_sizes=[(224, 224), (256, 256), (384, 384)],
    num_benchmark_runs=20
)

# Create benchmarker
benchmarker = ModelBenchmarker(config)

# Benchmark model
results = benchmarker.benchmark_single_model(
    model=model,
    model_name="resnet50",
    input_shape=(32, 3, 224, 224)
)
```

### Comprehensive Benchmarking

```python
# Define model configurations
model_configs = [
    {'name': 'ResNet50', 'type': 'cnn', 'params': {'model_name': 'resnet50'}},
    {'name': 'EfficientNet-B0', 'type': 'cnn', 'params': {'model_name': 'efficientnet_b0'}},
    {'name': 'ViT-Base', 'type': 'transformer', 'params': {'model_type': 'vit_base'}}
]

# Run comprehensive benchmark
results = benchmarker.benchmark_model_variants(model_configs)

# Generate report
report_path = benchmarker.generate_performance_report(results)
```

### Real-time Benchmarking

```python
from evaluation import RealTimeBenchmarker

# Create real-time benchmarker
rt_benchmarker = RealTimeBenchmarker(target_fps=30)

# Benchmark real-time performance
rt_results = rt_benchmarker.benchmark_realtime_performance(
    model=model,
    input_shape=(1, 3, 224, 224),
    device=device,
    num_frames=1000
)

print(f"Real-time FPS: {rt_results['actual_fps']:.1f}")
print(f"Suitable for real-time: {rt_results['is_realtime_suitable']}")
```

## Performance Analysis

### Model Analysis

```python
from evaluation import ModelAnalyzer

# Create analyzer
analyzer = ModelAnalyzer()

# Analyze model complexity
complexity = analyzer.analyze_model_complexity(model)
print(f"Total Parameters: {complexity['total_parameters']:,}")
print(f"Trainable Parameters: {complexity['trainable_parameters']:,}")

# Analyze prediction confidence
confidence_analysis = analyzer.analyze_prediction_confidence(
    model=model,
    dataloader=test_loader,
    device=device
)
```

### Feature Importance Analysis

```python
# Analyze feature importance
importance_analysis = analyzer.analyze_feature_importance(
    model=model,
    sample_input=sample_tensor,
    device=device
)

# Visualize importance
from evaluation import FeatureVisualizer
visualizer = FeatureVisualizer()
visualizer.plot_feature_importance(
    importance_analysis['gradient_mean'],
    feature_names=feature_names
)
```

### Dataset Analysis

```python
from data import DatasetAnalyzer

# Analyze dataset
dataset_analyzer = DatasetAnalyzer(dataset)
analysis_results = dataset_analyzer.analyze_dataset()

# Visualize analysis
dataset_analyzer.visualize_analysis("dataset_analysis.png")
```

## Real-time Evaluation

### Live Performance Monitoring

```python
from evaluation import RealTimeMetrics

# Initialize real-time metrics
rt_metrics = RealTimeMetrics()

# Update metrics for each frame
for frame in video_stream:
    start_time = time.time()
    
    # Run detection
    prediction = model(frame)
    
    # Update metrics
    rt_metrics.update_frame(
        frame_time=time.time(),
        detection_time=time.time() - start_time,
        prediction=prediction,
        target=ground_truth
    )
    
    # Get current performance
    current_fps = rt_metrics.get_fps()
    current_accuracy = rt_metrics.get_accuracy()
    
    print(f"FPS: {current_fps:.1f}, Accuracy: {current_accuracy:.3f}")
```

### Performance Visualization

```python
from evaluation import RealTimeVisualizer

# Create real-time visualizer
rt_visualizer = RealTimeVisualizer()

# Update with new data
rt_visualizer.update_buffer(
    prediction=prediction,
    confidence=confidence,
    fps=current_fps,
    latency=detection_time
)

# Create real-time plot
plot = rt_visualizer.create_realtime_plot()
```

## Best Practices

### Evaluation Protocol

1. **Train/Validation/Test Split**
   - Use 70/15/15 split
   - Ensure balanced classes
   - Stratified sampling

2. **Cross-Validation**
   - Use k-fold cross-validation
   - Report mean and standard deviation
   - Account for data leakage

3. **Statistical Significance**
   - Perform multiple runs
   - Report confidence intervals
   - Use appropriate statistical tests

### Performance Optimization

1. **Batch Processing**
   - Process multiple samples together
   - Optimize batch size for hardware
   - Use mixed precision training

2. **Memory Management**
   - Monitor memory usage
   - Use gradient checkpointing
   - Clear unused variables

3. **Hardware Utilization**
   - Use GPU acceleration
   - Optimize data loading
   - Parallel processing

### Reporting Results

1. **Comprehensive Metrics**
   - Report all relevant metrics
   - Include confidence intervals
   - Compare with baselines

2. **Visualization**
   - Create clear plots
   - Use consistent formatting
   - Include error bars

3. **Reproducibility**
   - Set random seeds
   - Document hyperparameters
   - Provide code and data

## Common Issues and Solutions

### Low Accuracy
- **Problem**: Model performs poorly
- **Solutions**:
  - Increase training data
  - Improve data quality
  - Adjust hyperparameters
  - Use data augmentation

### Slow Inference
- **Problem**: Model is too slow
- **Solutions**:
  - Use smaller model
  - Apply quantization
  - Optimize preprocessing
  - Use batch processing

### High Memory Usage
- **Problem**: Model uses too much memory
- **Solutions**:
  - Reduce batch size
  - Use gradient checkpointing
  - Apply model pruning
  - Use mixed precision

### Poor Generalization
- **Problem**: Model doesn't generalize well
- **Solutions**:
  - Improve data diversity
  - Use regularization
  - Apply domain adaptation
  - Use ensemble methods

## Advanced Evaluation Techniques

### Adversarial Evaluation

```python
# Test robustness to adversarial attacks
from evaluation import AdversarialEvaluator

adversarial_evaluator = AdversarialEvaluator()
robustness_results = adversarial_evaluator.evaluate_robustness(
    model=model,
    test_loader=test_loader,
    attack_types=['fgsm', 'pgd', 'cw']
)
```

### Cross-Domain Evaluation

```python
# Evaluate on different domains
cross_domain_results = {}
for domain in ['synthetic', 'real', 'compressed']:
    domain_dataset = load_domain_dataset(domain)
    results = evaluate_model(model, domain_dataset)
    cross_domain_results[domain] = results
```

### Ablation Studies

```python
# Study individual components
ablation_results = {}
for component in ['backbone', 'head', 'augmentation']:
    model_variant = create_model_variant(component)
    results = evaluate_model(model_variant, test_dataset)
    ablation_results[component] = results
```

## Conclusion

This evaluation guide provides comprehensive tools and techniques for evaluating deepfake detection models. By following these practices, researchers can ensure robust and reliable evaluation of their models.

For more advanced techniques and specific use cases, refer to the individual module documentation and research papers in the references section.
