"""
Train and evaluate all sentiment analysis models
Generates comprehensive results for class project
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

from src.utils.path_utils import get_path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.spark_config import create_spark_session
from src.ml.sentiment_models import ModelEvaluator
from src.ml.deep_learning_models import evaluate_deep_learning_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_traditional_models(spark, sample_size=1.0):
    """Train traditional ML models"""
    logger.info("Training traditional ML models...")

    # Load data
    df = spark.read.parquet(str(get_path("data/processed/pipeline_features")))
    df_sample = df.sample(sample_size)

    # Define features
    feature_cols = [
        "text_length", "processed_length", "token_count",
        "emoji_sentiment", "exclamation_count", "question_count",
        "uppercase_ratio", "punctuation_density",
        "vader_compound", "vader_positive", "vader_negative", "vader_neutral",
        "hour_sin", "hour_cos", "is_weekend",
        "2gram_count", "3gram_count"
    ]

    # Split data
    train_df, test_df = df_sample.randomSplit([0.8, 0.2], seed=42)

    # Cache for performance
    train_df.cache()
    test_df.cache()

    # Evaluate models
    evaluator = ModelEvaluator(spark)
    comparison_df = evaluator.evaluate_all_models(train_df, test_df, feature_cols)

    # Perform cross-validation on best model
    cv_results = evaluator.perform_cross_validation(
        df_sample, feature_cols, "Random Forest"
    )

    # Save results
    output_path = str(get_path("data/models"))
    os.makedirs(output_path, exist_ok=True)
    evaluator.save_results(comparison_df, output_path)

    # Clean up
    train_df.unpersist()
    test_df.unpersist()

    return comparison_df, cv_results


def train_deep_learning_models(spark, sample_size=1.0):
    """Train deep learning models"""
    logger.info("Training deep learning models...")

    results = evaluate_deep_learning_models(spark, sample_size)

    # Save results
    output_path = str(get_path("data/models/deep_learning_results.json"))
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results

# def deep_learning_models_results(spark, sample_size=1.0):
#     logger.info("Model results from deep learning models...")
#     return str(get_path("data/models/pipeline_results_colab.json"))


def generate_performance_report(traditional_results, dl_results, cv_results):
    """Generate comprehensive performance report"""
    logger.info("Generating performance report...")

    report = """
# Sentiment Analysis Model Performance Report

## Traditional Machine Learning Models

### Model Comparison
{}

### Cross-Validation Results
- Best F1 Score: {:.4f}
- CV Scores: {}

## Deep Learning Models

### LSTM Model
- Accuracy: {:.4f}
- AUC: {:.4f}

### CNN Model
- Accuracy: {:.4f}
- AUC: {:.4f}

### Transformer Model
- Accuracy: {:.4f}
- AUC: {:.4f}

## Best Performing Model
{}

## Recommendations
1. {} shows the best balance of accuracy and training time
2. For production use, consider ensemble methods combining top models
3. Deep learning models show promise but require more data for optimal performance
""".format(
        traditional_results.to_string(),
        cv_results['cv_best_score'],
        [f"{s:.4f}" for s in cv_results['cv_scores']],
        dl_results['LSTM']['accuracy'],
        dl_results['LSTM']['auc'],
        dl_results['CNN']['accuracy'],
        dl_results['CNN']['auc'],
        dl_results['Transformer']['accuracy'],
        dl_results['Transformer']['auc'],
        traditional_results.iloc[0].name,
        traditional_results.iloc[0].name
    )

    # Save report
    with open(str(get_path("data/models/performance_report.md")), 'w') as f:
        f.write(report)

    logger.info("Performance report saved")
    return report


def main():
    """Main training pipeline"""
    start_time = time.time()

    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs(str(get_path("data/models")), exist_ok=True)

    # Initialize Spark
    spark = create_spark_session("ModelTraining")

    try:
        # Train traditional models
        traditional_results, cv_results = train_traditional_models(spark, sample_size=1.0)

        # Train deep learning models
        dl_results = train_deep_learning_models(spark, sample_size=1.0)
        # dl_results = deep_learning_models_results(spark, sample_size=1.0)

        # Generate report
        report = generate_performance_report(traditional_results, dl_results, cv_results)

        # Log completion
        total_time = time.time() - start_time
        logger.info(f"\nTotal training time: {total_time:.2f} seconds")
        logger.info("All models trained successfully!")

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Results saved to: data/models/")
        print(f"Logs saved to: logs/model_training.log")
        print("\nTop 3 Models by F1 Score:")
        print(traditional_results.head(3)[['f1', 'accuracy', 'training_time']])

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()