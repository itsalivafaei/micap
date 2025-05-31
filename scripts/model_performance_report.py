"""
Module: generate_model_performance_report
Purpose: Generate comprehensive performance reports from saved models without dependencies
Author: AI Assistant
Date: 2024-01-20
Dependencies: pandas, numpy, matplotlib, seaborn, sklearn, json, logging

This script loads saved models and generates performance reports independently of the training pipeline.
It evaluates both traditional ML and deep learning models if available.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ML metrics and utilities
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.path_utils import get_path

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/performance_report.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ModelPerformanceAnalyzer:
    """
    Analyzes saved model performance and generates comprehensive reports.
    Works independently of training pipeline.
    """
    
    def __init__(self, model_dir: str = None, data_dir: str = None):
        """
        Initialize the performance analyzer.
        
        Args:
            model_dir: Directory containing saved models
            data_dir: Directory containing test data
        """
        self.model_dir = Path(model_dir or get_path("data/models"))
        self.data_dir = Path(data_dir or get_path("data/processed"))
        self.results = {}
        self.test_data = None
        self.visualizations_dir = Path(get_path("data/visualizations/performance"))
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ModelPerformanceAnalyzer")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Data directory: {self.data_dir}")
        
    def load_test_data(self, sample_fraction: float = 1.0) -> Tuple[Any, Any, List[str]]:
        """
        Load test data for model evaluation.
        
        Args:
            sample_fraction: Fraction of data to use for testing
            
        Returns:
            Tuple of (features, labels, feature_names)
        """
        logger.info("Loading test data...")
        
        try:
            # Try to load preprocessed data from parquet
            test_data_path = self.data_dir / "pipeline_features"
            
            if test_data_path.exists():
                # Use pandas for loading if Spark not available
                import pandas as pd
                
                # Load parquet files
                df = pd.read_parquet(test_data_path)
                logger.info(f"Loaded {len(df)} records from {test_data_path}")
                
                # Sample if requested
                if sample_fraction < 1.0:
                    df = df.sample(frac=sample_fraction, random_state=42)
                    logger.info(f"Sampled {len(df)} records ({sample_fraction*100}%)")
                
                # Extract features and labels
                feature_cols = [
                    "text_length", "processed_length", "token_count",
                    "emoji_sentiment", "exclamation_count", "question_count",
                    "uppercase_ratio", "punctuation_density",
                    "vader_compound", "vader_positive", "vader_negative", "vader_neutral",
                    "hour_sin", "hour_cos", "is_weekend",
                    "2gram_count", "3gram_count"
                ]
                
                # Filter available features
                available_features = [col for col in feature_cols if col in df.columns]
                logger.info(f"Using {len(available_features)} features: {available_features}")
                
                X = df[available_features].values
                y = df['sentiment'].values
                
                self.test_data = (X, y, available_features)
                return X, y, available_features
                
            else:
                logger.error(f"Test data not found at {test_data_path}")
                raise FileNotFoundError(f"Test data not found at {test_data_path}")
                
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            raise
            
    def load_saved_models(self) -> Dict[str, Any]:
        """
        Load all saved models from the model directory.
        
        Returns:
            Dictionary of loaded models
        """
        logger.info(f"Loading saved models from {self.model_dir}")
        models = {}
        
        # Load traditional ML model results from CSV/JSON if available
        try:
            # Check for model comparison results
            comparison_file = self.model_dir / "model_comparison.csv"
            if comparison_file.exists():
                comparison_df = pd.read_csv(comparison_file)
                logger.info(f"Loaded model comparison results: {len(comparison_df)} models")
                models['traditional_results'] = comparison_df
            
            # Check for deep learning results
            dl_results_file = self.model_dir / "deep_learning_results.json"
            if dl_results_file.exists():
                with open(dl_results_file, 'r') as f:
                    dl_results = json.load(f)
                logger.info(f"Loaded deep learning results: {len(dl_results)} models")
                models['deep_learning_results'] = dl_results
                
            # Check for benchmark results
            benchmark_file = self.model_dir / "benchmark_results.csv"
            if benchmark_file.exists():
                benchmark_df = pd.read_csv(benchmark_file)
                logger.info(f"Loaded benchmark results: {len(benchmark_df)} records")
                models['benchmark_results'] = benchmark_df
                
        except Exception as e:
            logger.error(f"Error loading model results: {e}")
            
        if not models:
            logger.warning("No saved model results found")
            
        return models
        
    def calculate_additional_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Add AUC if probabilities available
        if y_prob is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc'] = 0.0
                
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        # Calculate additional metrics
        total = len(y_true)
        metrics['specificity'] = metrics['true_negatives'] / (metrics['true_negatives'] + metrics['false_positives'])
        metrics['sensitivity'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
        metrics['balanced_accuracy'] = (metrics['specificity'] + metrics['sensitivity']) / 2
        
        return metrics
        
    def generate_performance_visualizations(self, results: Dict[str, Any]):
        """
        Generate comprehensive performance visualizations.
        
        Args:
            results: Dictionary containing model results
        """
        logger.info("Generating performance visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Model Comparison Bar Chart
        if 'traditional_results' in results:
            self._plot_model_comparison(results['traditional_results'])
            
        # 2. Training Time vs Accuracy Trade-off
        if 'traditional_results' in results:
            self._plot_time_accuracy_tradeoff(results['traditional_results'])
            
        # 3. Scalability Analysis
        if 'benchmark_results' in results:
            self._plot_scalability_analysis(results['benchmark_results'])
            
        # 4. Deep Learning vs Traditional ML Comparison
        if 'traditional_results' in results and 'deep_learning_results' in results:
            self._plot_ml_dl_comparison(results['traditional_results'], 
                                      results['deep_learning_results'])
            
        # 5. Comprehensive Metrics Heatmap
        self._plot_metrics_heatmap(results)
        
        logger.info(f"Visualizations saved to {self.visualizations_dir}")
        
    def _plot_model_comparison(self, df: pd.DataFrame):
        """Create model comparison bar chart."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        ax = axes[0, 0]
        models = df.index if isinstance(df.index, pd.Index) else df['model']
        ax.bar(models, df['accuracy'], color='skyblue', alpha=0.8)
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim(0, 1)
        for i, v in enumerate(df['accuracy']):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # F1 Score comparison
        ax = axes[0, 1]
        ax.bar(models, df['f1'], color='lightcoral', alpha=0.8)
        ax.set_ylabel('F1 Score')
        ax.set_title('Model F1 Score Comparison')
        ax.set_ylim(0, 1)
        for i, v in enumerate(df['f1']):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Training time comparison
        ax = axes[1, 0]
        ax.bar(models, df['training_time'], color='lightgreen', alpha=0.8)
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Model Training Time Comparison')
        for i, v in enumerate(df['training_time']):
            ax.text(i, v + 0.5, f'{v:.1f}s', ha='center')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # AUC comparison
        ax = axes[1, 1]
        ax.bar(models, df['auc'], color='gold', alpha=0.8)
        ax.set_ylabel('AUC Score')
        ax.set_title('Model AUC Comparison')
        ax.set_ylim(0, 1)
        for i, v in enumerate(df['auc']):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved model comparison visualization")
        
    def _plot_time_accuracy_tradeoff(self, df: pd.DataFrame):
        """Plot training time vs accuracy trade-off."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = df.index if isinstance(df.index, pd.Index) else df['model']
        scatter = ax.scatter(df['training_time'], df['accuracy'], 
                           s=df['f1']*500, alpha=0.6, c=range(len(df)), cmap='viridis')
        
        # Add model labels
        for i, model in enumerate(models):
            ax.annotate(model, (df.iloc[i]['training_time'], df.iloc[i]['accuracy']),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Training Time (seconds)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Time vs Accuracy Trade-off\n(Bubble size = F1 Score)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Model Index')
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'time_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved time-accuracy trade-off visualization")
        
    def _plot_scalability_analysis(self, df: pd.DataFrame):
        """Plot scalability analysis from benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training time scalability
        ax = axes[0, 0]
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax.plot(model_data['sample_count'], model_data['training_time'],
                   marker='o', label=model, linewidth=2)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Time Scalability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Prediction time scalability
        ax = axes[0, 1]
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax.plot(model_data['sample_count'], model_data['prediction_time'],
                   marker='s', label=model, linewidth=2)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Prediction Time (seconds)')
        ax.set_title('Prediction Time Scalability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy vs data size
        ax = axes[1, 0]
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax.plot(model_data['sample_count'], model_data['accuracy'],
                   marker='^', label=model, linewidth=2)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy vs Training Data Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.0)
        
        # Training efficiency (samples per second)
        ax = axes[1, 1]
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            efficiency = model_data['sample_count'] / model_data['training_time']
            ax.plot(model_data['sample_count'], efficiency,
                   marker='d', label=model, linewidth=2)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Training Efficiency (samples/second)')
        ax.set_title('Training Efficiency Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'scalability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved scalability analysis visualization")
        
    def _plot_ml_dl_comparison(self, traditional_df: pd.DataFrame, dl_results: Dict):
        """Compare traditional ML and deep learning models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        all_models = []
        all_accuracies = []
        all_f1s = []
        model_types = []
        
        # Traditional ML
        models = traditional_df.index if isinstance(traditional_df.index, pd.Index) else traditional_df['model']
        for i, model in enumerate(models):
            all_models.append(model)
            all_accuracies.append(traditional_df.iloc[i]['accuracy'])
            all_f1s.append(traditional_df.iloc[i]['f1'])
            model_types.append('Traditional ML')
        
        # Deep Learning
        for model_name, metrics in dl_results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                all_models.append(model_name)
                all_accuracies.append(metrics['accuracy'])
                all_f1s.append(metrics.get('f1', metrics['accuracy']))  # Use accuracy if F1 not available
                model_types.append('Deep Learning')
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': all_models,
            'Accuracy': all_accuracies,
            'F1_Score': all_f1s,
            'Type': model_types
        })
        
        # Accuracy comparison
        comparison_df.pivot(columns='Type', values='Accuracy').plot(kind='bar', ax=ax1)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Traditional ML vs Deep Learning - Accuracy')
        ax1.legend(title='Model Type')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # F1 Score comparison
        comparison_df.pivot(columns='Type', values='F1_Score').plot(kind='bar', ax=ax2)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Traditional ML vs Deep Learning - F1 Score')
        ax2.legend(title='Model Type')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'ml_dl_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved ML vs DL comparison visualization")
        
    def _plot_metrics_heatmap(self, results: Dict):
        """Create a comprehensive metrics heatmap."""
        # Prepare data for heatmap
        metrics_data = []
        model_names = []
        
        # Traditional ML metrics
        if 'traditional_results' in results:
            df = results['traditional_results']
            models = df.index if isinstance(df.index, pd.Index) else df['model']
            for i, model in enumerate(models):
                model_names.append(model)
                metrics_data.append([
                    df.iloc[i]['accuracy'],
                    df.iloc[i]['precision'],
                    df.iloc[i]['recall'],
                    df.iloc[i]['f1'],
                    df.iloc[i]['auc']
                ])
        
        # Deep Learning metrics
        if 'deep_learning_results' in results:
            for model_name, metrics in results['deep_learning_results'].items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    model_names.append(f"DL_{model_name}")
                    metrics_data.append([
                        metrics['accuracy'],
                        metrics.get('precision', metrics['accuracy']),
                        metrics.get('recall', metrics['accuracy']),
                        metrics.get('f1', metrics['accuracy']),
                        metrics.get('auc', 0.5)
                    ])
        
        if metrics_data:
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            metrics_df = pd.DataFrame(
                metrics_data,
                index=model_names,
                columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
            )
            
            sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd',
                       cbar_kws={'label': 'Score'}, ax=ax)
            ax.set_title('Model Performance Metrics Heatmap')
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Models')
            
            plt.tight_layout()
            plt.savefig(self.visualizations_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved metrics heatmap visualization")
            
    def generate_markdown_report(self, results: Dict) -> str:
        """
        Generate a comprehensive markdown report.
        
        Args:
            results: Dictionary containing all model results
            
        Returns:
            Markdown report as string
        """
        logger.info("Generating markdown report...")
        
        report = f"""# Model Performance Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report provides a comprehensive analysis of sentiment analysis model performance across traditional machine learning and deep learning approaches.

"""
        
        # Traditional ML Results
        if 'traditional_results' in results:
            df = results['traditional_results']
            best_model_idx = df['f1'].idxmax()
            best_model = df.index[best_model_idx] if isinstance(df.index, pd.Index) else df.iloc[best_model_idx]['model']
            
            report += f"""## Traditional Machine Learning Models

### Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score | AUC | Training Time (s) |
|-------|----------|-----------|--------|----------|-----|------------------|
"""
            models = df.index if isinstance(df.index, pd.Index) else df['model']
            for i, model in enumerate(models):
                report += f"| {model} | {df.iloc[i]['accuracy']:.4f} | {df.iloc[i]['precision']:.4f} | "
                report += f"{df.iloc[i]['recall']:.4f} | {df.iloc[i]['f1']:.4f} | {df.iloc[i]['auc']:.4f} | "
                report += f"{df.iloc[i]['training_time']:.2f} |\n"
            
            report += f"""
### Best Performing Model
- **Model**: {best_model}
- **F1 Score**: {df.iloc[best_model_idx]['f1']:.4f}
- **Accuracy**: {df.iloc[best_model_idx]['accuracy']:.4f}
- **Training Time**: {df.iloc[best_model_idx]['training_time']:.2f} seconds

"""
        
        # Deep Learning Results
        if 'deep_learning_results' in results:
            report += """## Deep Learning Models

### Performance Summary

| Model | Accuracy | AUC | Loss |
|-------|----------|-----|------|
"""
            for model_name, metrics in results['deep_learning_results'].items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    report += f"| {model_name} | {metrics['accuracy']:.4f} | "
                    report += f"{metrics.get('auc', 'N/A'):.4f} | {metrics.get('loss', 'N/A'):.4f} |\n"
            
            report += "\n"
        
        # Scalability Analysis
        if 'benchmark_results' in results:
            df = results['benchmark_results']
            report += """## Scalability Analysis

### Training Time Scaling
"""
            # Group by model and show scaling
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                report += f"\n**{model}**:\n"
                for _, row in model_data.iterrows():
                    report += f"- {row['sample_count']:,} samples: {row['training_time']:.2f}s "
                    report += f"(Accuracy: {row['accuracy']:.4f})\n"
            
            report += "\n"
        
        # Key Insights
        report += """## Key Insights and Recommendations

### Performance Analysis
"""
        
        if 'traditional_results' in results:
            df = results['traditional_results']
            
            # Find fastest accurate model
            accurate_models = df[df['f1'] > df['f1'].mean()]
            if not accurate_models.empty:
                fastest_idx = accurate_models['training_time'].idxmin()
                fastest_model = accurate_models.index[fastest_idx] if isinstance(accurate_models.index, pd.Index) else accurate_models.iloc[fastest_idx]['model']
                
                report += f"""
1. **Best Overall Model**: {best_model} with F1 score of {df.iloc[best_model_idx]['f1']:.4f}
2. **Fastest Accurate Model**: {fastest_model} achieves {accurate_models.iloc[fastest_idx]['f1']:.4f} F1 score in just {accurate_models.iloc[fastest_idx]['training_time']:.2f} seconds
3. **Trade-off Analysis**: 
   - For real-time applications: Consider {fastest_model}
   - For maximum accuracy: Use {best_model}
"""
        
        # Scalability insights
        if 'benchmark_results' in results:
            report += """
### Scalability Insights

Based on the benchmark analysis:
"""
            df = results['benchmark_results']
            
            # Calculate scaling efficiency
            for model in df['model'].unique():
                model_data = df[df['model'] == model].sort_values('sample_count')
                if len(model_data) > 1:
                    # Calculate time complexity (roughly)
                    sizes = model_data['sample_count'].values
                    times = model_data['training_time'].values
                    
                    # Simple linear regression for scaling
                    if len(sizes) > 1:
                        slope = np.polyfit(sizes, times, 1)[0]
                        report += f"- **{model}**: ~{slope:.6f} seconds per sample (linear scaling)\n"
        
        report += """
## Visualizations

The following visualizations have been generated:
1. Model Comparison Charts (`model_comparison.png`)
2. Time-Accuracy Trade-off Analysis (`time_accuracy_tradeoff.png`)
3. Scalability Analysis (`scalability_analysis.png`)
4. ML vs DL Comparison (`ml_dl_comparison.png`)
5. Performance Metrics Heatmap (`metrics_heatmap.png`)

## Conclusion

This comprehensive analysis provides insights into model performance, scalability, and practical deployment considerations. The visualizations and metrics enable informed decision-making for production deployment.

---
*Report generated by MICAP Performance Analyzer*
"""
        
        return report
        
    def generate_full_report(self, output_path: Optional[str] = None):
        """
        Generate complete performance report with all analyses.
        
        Args:
            output_path: Optional path to save the report
        """
        logger.info("Starting full performance report generation...")
        
        try:
            # Load saved models and results
            results = self.load_saved_models()
            
            if not results:
                logger.error("No saved model results found. Please train models first.")
                return
            
            # Generate visualizations
            self.generate_performance_visualizations(results)
            
            # Generate markdown report
            report = self.generate_markdown_report(results)
            
            # Save report
            if output_path is None:
                output_path = self.model_dir / "performance_report_comprehensive.md"
            
            with open(output_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Performance report saved to: {output_path}")
            
            # Also save as HTML for better viewing
            try:
                import markdown
                html_content = markdown.markdown(report, extensions=['tables', 'fenced_code'])
                html_path = str(output_path).replace('.md', '.html')
                
                with open(html_path, 'w') as f:
                    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; overflow-x: auto; }}
        h1, h2, h3 {{ color: #333; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #666; padding-bottom: 5px; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>""")
                logger.info(f"HTML report saved to: {html_path}")
            except ImportError:
                logger.warning("Markdown module not available for HTML conversion")
            
            # Print summary to console
            print("\n" + "="*60)
            print("PERFORMANCE REPORT GENERATION COMPLETE")
            print("="*60)
            print(f"Report saved to: {output_path}")
            print(f"Visualizations saved to: {self.visualizations_dir}")
            
            if 'traditional_results' in results:
                print("\nTop 3 Models by F1 Score:")
                df = results['traditional_results']
                print(df.nlargest(3, 'f1')[['accuracy', 'f1', 'training_time']])
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)
            raise


def main():
    """
    Main function to generate performance reports from saved models.
    """
    logger.info("="*60)
    logger.info("MODEL PERFORMANCE REPORT GENERATOR")
    logger.info("="*60)
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Initialize analyzer
        analyzer = ModelPerformanceAnalyzer()
        
        # Generate comprehensive report
        report = analyzer.generate_full_report()
        
        logger.info("Performance report generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()