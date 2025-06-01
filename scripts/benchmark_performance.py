"""
Performance benchmarking for scalability analysis
Tests models with increasing data sizes.
."""

import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.path_utils import get_path

sys.path.append(str(Path(__file__).parent.parent))

from config.spark_config import create_spark_session
from src.ml.sentiment_models import (
    NaiveBayesModel, LogisticRegressionModel,
    RandomForestModel, GradientBoostingModel
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_models(spark, data_sizes=[0.01, 0.05, 0.1, 0.2]):
    """
    Benchmark model performance with different data sizes.
    ."""
    logger.info("Starting performance benchmarking...")

    # Load full dataset

    df = spark.read.parquet(str(get_path("data/processed/pipeline_features")))

    # Define features
    feature_cols = [
        "text_length", "processed_length", "token_count",
        "emoji_sentiment", "exclamation_count", "question_count",
        "uppercase_ratio", "punctuation_density",
        "vader_compound", "vader_positive", "vader_negative", "vader_neutral",
        "hour_sin", "hour_cos", "is_weekend",
        "2gram_count", "3gram_count"
    ]

    # Models to benchmark
    models = [
        ("Naive Bayes", NaiveBayesModel),
        ("Logistic Regression", LogisticRegressionModel),
        ("Random Forest", RandomForestModel),
        ("Gradient Boosting", GradientBoostingModel)
    ]

    results = []

    for size in data_sizes:
        logger.info(f"\nBenchmarking with {size * 100}% of data...")

        # Sample data
        df_sample = df.sample(size, seed=42)
        sample_count = df_sample.count()

        # Split data
        train_df, test_df = df_sample.randomSplit([0.8, 0.2], seed=42)

        # Cache for performance
        train_df.cache()
        test_df.cache()

        for model_name, ModelClass in models:
            logger.info(f"Testing {model_name}...")

            # Initialize model
            model = ModelClass(spark)

            # Measure training time
            start_time = time.time()
            model.train(train_df, feature_cols)
            training_time = time.time() - start_time

            # Measure prediction time
            start_time = time.time()
            metrics = model.evaluate(test_df, feature_cols)
            prediction_time = time.time() - start_time

            # Store results
            results.append({
                'model': model_name,
                'data_size': size,
                'sample_count': sample_count,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1']
            })

            logger.info(f"  Training time: {training_time:.2f}s")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")

        # Unpersist
        train_df.unpersist()
        test_df.unpersist()

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(str(get_path("data/models/benchmark_results.csv")), index=False)

    return results_df


def visualize_benchmarks(results_df):
    """Create benchmark visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Training time vs data size
    ax = axes[0, 0]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        ax.plot(model_data['sample_count'], model_data['training_time'],
                marker='o', label=model)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Scalability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Prediction time vs data size
    ax = axes[0, 1]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        ax.plot(model_data['sample_count'], model_data['prediction_time'],
                marker='s', label=model)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Prediction Time (seconds)')
    ax.set_title('Prediction Time Scalability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy vs data size
    ax = axes[1, 0]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        ax.plot(model_data['sample_count'], model_data['accuracy'],
                marker='^', label=model)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Training Data Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1 score vs data size
    ax = axes[1, 1]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        ax.plot(model_data['sample_count'], model_data['f1'],
                marker='d', label=model)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Training Data Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(get_path("data/models/benchmark_plots.png")), dpi=300)
    plt.show()


def main():
    """Run benchmarking."""
    spark = create_spark_session("Benchmarking")

    # Run benchmarks
    results_df = benchmark_models(spark, data_sizes=[0.01, 0.05, 0.1, 0.2])

    # Visualize results
    visualize_benchmarks(results_df)

    # Print summary
    print("\nBenchmark Summary:")
    print("=" * 60)
    print(results_df.groupby('model').agg({
        'training_time': 'mean',
        'prediction_time': 'mean',
        'accuracy': 'mean',
        'f1': 'mean'
    }).round(4))

    spark.stop()


if __name__ == "__main__":
    main()