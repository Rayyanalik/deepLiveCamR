"""
Deepfake Evasion Testing Module

This module tests how well generated deepfakes can evade detection algorithms.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from .deepfake_detector import DeepfakeDetector
from .feature_analyzer import FeatureAnalyzer


class EvasionTester:
    """
    Tests deepfake evasion capabilities against various detection methods.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the evasion tester.
        
        Args:
            device: Device to run inference on
        """
        self.detector = DeepfakeDetector(device=device)
        self.feature_analyzer = FeatureAnalyzer()
        
        # Evasion techniques
        self.evasion_techniques = {
            'blur': self._apply_blur,
            'noise': self._apply_noise,
            'compression': self._apply_compression,
            'color_adjustment': self._apply_color_adjustment,
            'contrast_adjustment': self._apply_contrast_adjustment,
            'brightness_adjustment': self._apply_brightness_adjustment,
            'gaussian_blur': self._apply_gaussian_blur,
            'motion_blur': self._apply_motion_blur
        }
        
        # Test results storage
        self.test_results = {}
        
    def _apply_blur(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply blur to reduce detection artifacts."""
        kernel_size = int(1 + intensity * 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.blur(image, (kernel_size, kernel_size))
    
    def _apply_noise(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply noise to mask artifacts."""
        noise = np.random.normal(0, intensity * 25, image.shape).astype(np.int16)
        noisy_image = image.astype(np.int16) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def _apply_compression(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply JPEG-like compression artifacts."""
        quality = int(100 - intensity * 80)  # 20-100 quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encoded_img, 1)
    
    def _apply_color_adjustment(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply color space adjustments."""
        # Convert to HSV and adjust saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * (1 + intensity * 0.5)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def _apply_contrast_adjustment(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply contrast adjustment."""
        alpha = 1 + intensity * 0.5  # Contrast control
        beta = intensity * 50  # Brightness control
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def _apply_brightness_adjustment(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply brightness adjustment."""
        beta = intensity * 100 - 50  # Brightness control
        return cv2.convertScaleAbs(image, beta=beta)
    
    def _apply_gaussian_blur(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply Gaussian blur."""
        kernel_size = int(1 + intensity * 15)
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = intensity * 5
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def _apply_motion_blur(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply motion blur."""
        kernel_size = int(5 + intensity * 20)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Apply motion blur
        motion_blurred = cv2.filter2D(image, -1, kernel)
        return motion_blurred
    
    def test_single_technique(self, image: np.ndarray, technique: str, 
                            intensity: float = 0.5) -> Dict[str, float]:
        """
        Test a single evasion technique.
        
        Args:
            image: Input image
            technique: Evasion technique name
            intensity: Technique intensity (0-1)
            
        Returns:
            Detection results
        """
        if technique not in self.evasion_techniques:
            raise ValueError(f"Unknown technique: {technique}")
        
        # Apply evasion technique
        modified_image = self.evasion_techniques[technique](image, intensity)
        
        # Test detection
        original_result = self.detector.detect_in_real_time(image)
        modified_result = self.detector.detect_in_real_time(modified_image)
        
        # Calculate evasion effectiveness
        original_score = original_result['overall_score']
        modified_score = modified_result['overall_score']
        evasion_effectiveness = max(0, original_score - modified_score)
        
        return {
            'technique': technique,
            'intensity': intensity,
            'original_score': original_score,
            'modified_score': modified_score,
            'evasion_effectiveness': evasion_effectiveness,
            'evasion_success': evasion_effectiveness > 0.1
        }
    
    def test_all_techniques(self, image: np.ndarray, 
                           intensities: List[float] = [0.2, 0.5, 0.8]) -> Dict[str, List[Dict]]:
        """
        Test all evasion techniques at different intensities.
        
        Args:
            image: Input image
            intensities: List of intensity values to test
            
        Returns:
            Results for all techniques
        """
        results = {}
        
        print(f"Testing {len(self.evasion_techniques)} techniques at {len(intensities)} intensities each...")
        
        for technique in self.evasion_techniques:
            print(f"Testing technique: {technique}")
            technique_results = []
            
            for intensity in intensities:
                try:
                    result = self.test_single_technique(image, technique, intensity)
                    technique_results.append(result)
                    
                    print(f"  Intensity {intensity:.1f}: "
                          f"Original={result['original_score']:.3f}, "
                          f"Modified={result['modified_score']:.3f}, "
                          f"Evasion={result['evasion_effectiveness']:.3f}")
                    
                except Exception as e:
                    print(f"  Intensity {intensity:.1f}: Error - {e}")
            
            results[technique] = technique_results
        
        self.test_results = results
        return results
    
    def find_best_evasion_combination(self, image: np.ndarray, 
                                     max_techniques: int = 3) -> Dict[str, any]:
        """
        Find the best combination of evasion techniques.
        
        Args:
            image: Input image
            max_techniques: Maximum number of techniques to combine
            
        Returns:
            Best combination results
        """
        print("Finding best evasion combination...")
        
        # Test individual techniques first
        individual_results = self.test_all_techniques(image)
        
        # Find best individual techniques
        best_techniques = []
        for technique, results in individual_results.items():
            if results:
                best_result = max(results, key=lambda x: x['evasion_effectiveness'])
                best_techniques.append((technique, best_result))
        
        # Sort by effectiveness
        best_techniques.sort(key=lambda x: x[1]['evasion_effectiveness'], reverse=True)
        
        # Test combinations
        best_combination = None
        best_effectiveness = 0
        
        # Test top individual techniques
        for i in range(min(max_techniques, len(best_techniques))):
            technique, result = best_techniques[i]
            
            if result['evasion_effectiveness'] > best_effectiveness:
                best_effectiveness = result['evasion_effectiveness']
                best_combination = {
                    'techniques': [technique],
                    'effectiveness': result['evasion_effectiveness'],
                    'original_score': result['original_score'],
                    'modified_score': result['modified_score']
                }
        
        # Test 2-technique combinations
        if len(best_techniques) >= 2:
            for i in range(len(best_techniques)):
                for j in range(i + 1, len(best_techniques)):
                    tech1, result1 = best_techniques[i]
                    tech2, result2 = best_techniques[j]
                    
                    # Apply both techniques
                    modified_image = image.copy()
                    modified_image = self.evasion_techniques[tech1](
                        modified_image, result1['intensity'])
                    modified_image = self.evasion_techniques[tech2](
                        modified_image, result2['intensity'])
                    
                    # Test detection
                    modified_result = self.detector.detect_in_real_time(modified_image)
                    original_score = result1['original_score']
                    modified_score = modified_result['overall_score']
                    effectiveness = max(0, original_score - modified_score)
                    
                    if effectiveness > best_effectiveness:
                        best_effectiveness = effectiveness
                        best_combination = {
                            'techniques': [tech1, tech2],
                            'effectiveness': effectiveness,
                            'original_score': original_score,
                            'modified_score': modified_score
                        }
        
        return best_combination
    
    def generate_evasion_report(self, results: Dict[str, List[Dict]]) -> str:
        """
        Generate a comprehensive evasion report.
        
        Args:
            results: Test results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("DEEPFAKE EVASION TESTING REPORT")
        report.append("=" * 60)
        
        # Overall statistics
        all_effectiveness = []
        successful_techniques = 0
        
        for technique, technique_results in results.items():
            if technique_results:
                effectiveness_scores = [r['evasion_effectiveness'] for r in technique_results]
                all_effectiveness.extend(effectiveness_scores)
                
                if any(r['evasion_success'] for r in technique_results):
                    successful_techniques += 1
        
        if all_effectiveness:
            avg_effectiveness = np.mean(all_effectiveness)
            max_effectiveness = np.max(all_effectiveness)
            min_effectiveness = np.min(all_effectiveness)
            
            report.append(f"\n📊 OVERALL STATISTICS:")
            report.append(f"  Average Evasion Effectiveness: {avg_effectiveness:.3f}")
            report.append(f"  Maximum Evasion Effectiveness: {max_effectiveness:.3f}")
            report.append(f"  Minimum Evasion Effectiveness: {min_effectiveness:.3f}")
            report.append(f"  Successful Techniques: {successful_techniques}/{len(results)}")
        
        # Individual technique results
        report.append(f"\n🔍 INDIVIDUAL TECHNIQUE RESULTS:")
        
        for technique, technique_results in results.items():
            if not technique_results:
                continue
                
            report.append(f"\n  {technique.upper()}:")
            
            for result in technique_results:
                status = "✅ SUCCESS" if result['evasion_success'] else "❌ FAILED"
                report.append(f"    Intensity {result['intensity']:.1f}: "
                            f"Effectiveness={result['evasion_effectiveness']:.3f} {status}")
        
        # Best techniques
        report.append(f"\n🏆 BEST EVASION TECHNIQUES:")
        
        technique_rankings = []
        for technique, technique_results in results.items():
            if technique_results:
                best_result = max(technique_results, key=lambda x: x['evasion_effectiveness'])
                technique_rankings.append((technique, best_result))
        
        technique_rankings.sort(key=lambda x: x[1]['evasion_effectiveness'], reverse=True)
        
        for i, (technique, result) in enumerate(technique_rankings[:5]):
            report.append(f"  {i+1}. {technique}: {result['evasion_effectiveness']:.3f}")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        
        if successful_techniques > 0:
            report.append("  • Some techniques successfully reduce detection scores")
            report.append("  • Consider combining multiple techniques for better evasion")
            report.append("  • Focus on techniques with highest effectiveness scores")
        else:
            report.append("  • No techniques significantly reduced detection scores")
            report.append("  • Detection algorithm appears robust against tested methods")
            report.append("  • Consider more sophisticated evasion techniques")
        
        report.append("\n⚠️  IMPORTANT NOTES:")
        report.append("  • These tests are for research purposes only")
        report.append("  • Evasion techniques may reduce video quality")
        report.append("  • Detection algorithms continue to improve")
        report.append("  • Always follow ethical guidelines")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def plot_evasion_results(self, results: Dict[str, List[Dict]]):
        """
        Plot evasion test results.
        
        Args:
            results: Test results
        """
        # Prepare data for plotting
        techniques = []
        intensities = []
        effectiveness_scores = []
        
        for technique, technique_results in results.items():
            for result in technique_results:
                techniques.append(technique)
                intensities.append(result['intensity'])
                effectiveness_scores.append(result['evasion_effectiveness'])
        
        if not techniques:
            print("No data to plot")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Effectiveness by technique
        technique_means = {}
        for technique, technique_results in results.items():
            if technique_results:
                effectiveness_scores = [r['evasion_effectiveness'] for r in technique_results]
                technique_means[technique] = np.mean(effectiveness_scores)
        
        if technique_means:
            techniques_sorted = sorted(technique_means.items(), key=lambda x: x[1], reverse=True)
            tech_names, tech_scores = zip(*techniques_sorted)
            
            axes[0, 0].bar(range(len(tech_names)), tech_scores)
            axes[0, 0].set_xticks(range(len(tech_names)))
            axes[0, 0].set_xticklabels(tech_names, rotation=45)
            axes[0, 0].set_ylabel('Average Evasion Effectiveness')
            axes[0, 0].set_title('Evasion Effectiveness by Technique')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Effectiveness vs Intensity
        axes[0, 1].scatter(intensities, effectiveness_scores, alpha=0.6)
        axes[0, 1].set_xlabel('Technique Intensity')
        axes[0, 1].set_ylabel('Evasion Effectiveness')
        axes[0, 1].set_title('Evasion Effectiveness vs Intensity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Success rate by technique
        success_rates = {}
        for technique, technique_results in results.items():
            if technique_results:
                successful = sum(1 for r in technique_results if r['evasion_success'])
                success_rates[technique] = successful / len(technique_results)
        
        if success_rates:
            tech_names, rates = zip(*success_rates.items())
            axes[1, 0].bar(range(len(tech_names)), rates)
            axes[1, 0].set_xticks(range(len(tech_names)))
            axes[1, 0].set_xticklabels(tech_names, rotation=45)
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].set_title('Success Rate by Technique')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Score reduction distribution
        score_reductions = []
        for technique_results in results.values():
            for result in technique_results:
                reduction = result['original_score'] - result['modified_score']
                score_reductions.append(reduction)
        
        if score_reductions:
            axes[1, 1].hist(score_reductions, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Score Reduction')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Score Reductions')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
