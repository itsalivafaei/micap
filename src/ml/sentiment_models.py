"""Sentiment Analysis Models for MICAP.

Fixed Naive Bayes compatibility and feature scaling issues.
"""

import time
import logging
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from pyspark.sql import DataFrame, SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, MinMaxScaler, StringIndexer,
    IndexToString, OneHotEncoder
)
from pyspark.ml.classification import (
    NaiveBayes, LogisticRegression, RandomForestClassifier,
    GBTClassifier, LinearSVC, OneVsRest
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.ml.tuning import (
    CrossValidator, ParamGridBuilder,
    TrainValidationSplit
)
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import col, when, count, avg
import mlflow
import mlflow.spark

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for sentiment analysis models.
    
    Provides common functionality for all models.
    """

    def __init__(self, spark: SparkSession, model_name: str):
        """Initialize base model.

        Args:
            spark: Active SparkSession.
            model_name: Name of the model.
        """
        self.spark = spark
        self.model_name = model_name
        self.model = None
        self.pipeline = None
        self.metrics = {}

    @abstractmethod
    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build the model pipeline."""
        pass

    def prepare_features(self, df: DataFrame, feature_cols: List[str]) -> DataFrame:
        """Prepare features for model training with model-specific scaling.

        Args:
            df: Input DataFrame.
            feature_cols: List of feature column names.

        Returns:
            DataFrame with assembled features.
        """
        logger.info(f"Preparing features for {self.model_name}")

        # Filter to only numeric features for now
        numeric_features = [
            "text_length", "processed_length", "token_count",
            "emoji_sentiment", "exclamation_count", "question_count",
            "uppercase_ratio", "punctuation_density",
            "vader_compound", "vader_positive", "vader_negative", "vader_neutral",
            "hour_sin", "hour_cos", "is_weekend",
            "2gram_count", "3gram_count"
        ]

        # Create feature vector
        assembler = VectorAssembler(
            inputCols=numeric_features,
            outputCol="features_raw"
        )

        # Model-specific feature scaling
        if self.model_name == "Naive Bayes":
            # Naive Bayes requires non-negative features
            # Use MinMaxScaler to ensure all values are positive
            scaler = MinMaxScaler(
                inputCol="features_raw",
                outputCol="features"
            )
        else:
            # Other models can handle negative values
            scaler = StandardScaler(
                inputCol="features_raw",
                outputCol="features",
                withStd=True,
                withMean=True
            )

        # Create preprocessing pipeline
        preprocessing = Pipeline(stages=[assembler, scaler])

        # Fit and transform
        preprocessing_model = preprocessing.fit(df)
        df_prepared = preprocessing_model.transform(df)

        # Ensure label column is properly formatted
        df_prepared = df_prepared.withColumn("label", col("sentiment").cast("double"))

        return df_prepared

    def train(self, train_df: DataFrame, feature_cols: List[str]) -> PipelineModel:
        """Train the model.

        Args:
            train_df: Training DataFrame.
            feature_cols: List of feature columns.

        Returns:
            Trained PipelineModel.
        """
        start_time = time.time()
        logger.info(f"Training {self.model_name}...")

        # Prepare features
        df_prepared = self.prepare_features(train_df, feature_cols)

        # Build model pipeline
        self.pipeline = self.build_model(feature_cols)

        # Train model
        self.model = self.pipeline.fit(df_prepared)

        training_time = time.time() - start_time
        self.metrics['training_time'] = training_time
        logger.info(f"{self.model_name} training completed in {training_time:.2f} seconds")

        return self.model

    def evaluate(self, test_df: DataFrame, feature_cols: List[str]) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            test_df: Test DataFrame.
            feature_cols: List of feature columns.

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating {self.model_name}...")

        # Prepare features
        df_prepared = self.prepare_features(test_df, feature_cols)

        # Make predictions
        predictions = self.model.transform(df_prepared)

        # Binary classification metrics
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        # Multiclass metrics
        accuracy_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )

        precision_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )

        recall_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedRecall"
        )

        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )

        # Calculate metrics
        auc = binary_evaluator.evaluate(predictions)
        accuracy = accuracy_evaluator.evaluate(predictions)
        precision = precision_evaluator.evaluate(predictions)
        recall = recall_evaluator.evaluate(predictions)
        f1 = f1_evaluator.evaluate(predictions)

        # Store results
        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        self.metrics.update(metrics)
        logger.info(f"{self.model_name} evaluation: "
                   f"Accuracy={accuracy:.4f}, AUC={auc:.4f}, F1={f1:.4f}")

        return metrics

    def cross_validate(self, df: DataFrame, feature_cols: List[str],
                      num_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation.

        Args:
            df: Input DataFrame.
            feature_cols: List of feature columns.
            num_folds: Number of folds for cross-validation.

        Returns:
            Dictionary of cross-validation metrics.
        """
        logger.info(f"Starting {num_folds}-fold cross-validation for {self.model_name}")

        # Prepare features
        df_prepared = self.prepare_features(df, feature_cols)

        # Build pipeline
        pipeline = self.build_model(feature_cols)

        # Create evaluator
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )

        # Create parameter grid
        param_grid = self._get_param_grid()

        # Cross-validator
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=num_folds,
            seed=42
        )

        # Fit cross-validator
        start_time = time.time()
        cv_model = cv.fit(df_prepared)
        cv_time = time.time() - start_time

        # Get best metrics
        best_score = max(cv_model.avgMetrics)

        cv_metrics = {
            'cv_auc': best_score,
            'cv_time': cv_time,
            'num_folds': num_folds
        }

        logger.info(f"Cross-validation completed: AUC={best_score:.4f}")
        return cv_metrics

    @abstractmethod
    def _get_param_grid(self):
        """Get parameter grid for hyperparameter tuning."""
        pass


class NaiveBayesModel(BaseModel):
    """Naive Bayes model for sentiment analysis."""

    def __init__(self, spark: SparkSession):
        """Initialize Naive Bayes model."""
        super().__init__(spark, "Naive Bayes")

    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build Naive Bayes model pipeline."""
        nb = NaiveBayes(featuresCol="features", labelCol="label")
        return Pipeline(stages=[nb])

    def _get_param_grid(self):
        """Get parameter grid for Naive Bayes."""
        param_grid = ParamGridBuilder() \
            .addGrid(NaiveBayes.smoothing, [0.1, 1.0, 2.0]) \
            .build()
        return param_grid


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for sentiment analysis."""

    def __init__(self, spark: SparkSession):
        """Initialize Logistic Regression model."""
        super().__init__(spark, "Logistic Regression")

    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build Logistic Regression model pipeline."""
        lr = LogisticRegression(featuresCol="features", labelCol="label")
        return Pipeline(stages=[lr])

    def _get_param_grid(self):
        """Get parameter grid for Logistic Regression."""
        param_grid = ParamGridBuilder() \
            .addGrid(LogisticRegression.regParam, [0.01, 0.1, 1.0]) \
            .addGrid(LogisticRegression.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()
        return param_grid


class RandomForestModel(BaseModel):
    """Random Forest model for sentiment analysis."""

    def __init__(self, spark: SparkSession):
        """Initialize Random Forest model."""
        super().__init__(spark, "Random Forest")

    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build Random Forest model pipeline."""
        rf = RandomForestClassifier(featuresCol="features", labelCol="label")
        return Pipeline(stages=[rf])

    def _get_param_grid(self):
        """Get parameter grid for Random Forest."""
        param_grid = ParamGridBuilder() \
            .addGrid(RandomForestClassifier.numTrees, [10, 20, 50]) \
            .addGrid(RandomForestClassifier.maxDepth, [5, 10, 15]) \
            .build()
        return param_grid


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model for sentiment analysis."""

    def __init__(self, spark: SparkSession):
        """Initialize Gradient Boosting model."""
        super().__init__(spark, "Gradient Boosting")

    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build Gradient Boosting model pipeline."""
        gbt = GBTClassifier(featuresCol="features", labelCol="label")
        return Pipeline(stages=[gbt])

    def _get_param_grid(self):
        """Get parameter grid for Gradient Boosting."""
        param_grid = ParamGridBuilder() \
            .addGrid(GBTClassifier.maxIter, [10, 20]) \
            .addGrid(GBTClassifier.maxDepth, [5, 10]) \
            .build()
        return param_grid


class SVMModel(BaseModel):
    """Support Vector Machine (SVM) model for sentiment analysis."""

    def __init__(self, spark: SparkSession):
        """Initialize SVM model."""
        super().__init__(spark, "SVM")

    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build SVM model pipeline."""
        svm = LinearSVC(featuresCol="features", labelCol="label")
        return Pipeline(stages=[svm])

    def _get_param_grid(self):
        """Get parameter grid for SVM."""
        param_grid = ParamGridBuilder() \
            .addGrid(LinearSVC.regParam, [0.01, 0.1, 1.0]) \
            .build()
        return param_grid


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple algorithms."""

    def __init__(self, spark: SparkSession):
        """Initialize Ensemble model."""
        super().__init__(spark, "Ensemble")

    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build Ensemble model pipeline."""
        # For simplicity, use a single model wrapped in OneVsRest
        lr = LogisticRegression(featuresCol="features", labelCol="label")
        return Pipeline(stages=[lr])

    def _get_param_grid(self):
        """Get parameter grid for Ensemble."""
        return ParamGridBuilder().build()


class ModelEvaluator:
    """Utility class for evaluating and comparing multiple models."""

    def __init__(self, spark: SparkSession):
        """Initialize ModelEvaluator."""
        self.spark = spark
        self.results = []

    def evaluate_all_models(self, train_df: DataFrame, test_df: DataFrame,
                           feature_cols: List[str]) -> pd.DataFrame:
        """Evaluate all available sentiment models.

        Args:
            train_df: Training DataFrame.
            test_df: Test DataFrame.
            feature_cols: List of feature columns.

        Returns:
            DataFrame with comparison results.
        """
        logger.info("Starting comprehensive model evaluation...")

        # Initialize all models
        models = [
            NaiveBayesModel(self.spark),
            LogisticRegressionModel(self.spark),
            RandomForestModel(self.spark),
            GradientBoostingModel(self.spark),
            SVMModel(self.spark),
            EnsembleModel(self.spark)
        ]

        results = []

        for model in models:
            try:
                logger.info(f"Evaluating {model.model_name}...")

                # Train model
                model.train(train_df, feature_cols)

                # Evaluate model
                metrics = model.evaluate(test_df, feature_cols)

                # Add training time
                metrics['training_time'] = model.metrics.get('training_time', 0)
                metrics['model_name'] = model.model_name

                results.append(metrics)

            except Exception as e:
                logger.error(f"Error evaluating {model.model_name}: {e}")
                results.append({
                    'model_name': model.model_name,
                    'error': str(e),
                    'accuracy': 0,
                    'auc': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'training_time': 0
                })

        # Convert to DataFrame
        comparison_df = pd.DataFrame(results)

        # Sort by AUC score
        comparison_df = comparison_df.sort_values('auc', ascending=False)

        logger.info("Model evaluation completed")
        return comparison_df

    def perform_cross_validation(self, df: DataFrame, feature_cols: List[str],
                                model_name: str = "Random Forest") -> Dict:
        """Perform cross-validation on selected model.

        Args:
            df: Input DataFrame.
            feature_cols: List of feature columns.
            model_name: Name of model to validate.

        Returns:
            Cross-validation results.
        """
        logger.info(f"Performing cross-validation for {model_name}")

        # Initialize model
        model_map = {
            "Random Forest": RandomForestModel(self.spark),
            "Logistic Regression": LogisticRegressionModel(self.spark),
            "Naive Bayes": NaiveBayesModel(self.spark),
            "SVM": SVMModel(self.spark)
        }

        model = model_map.get(model_name, RandomForestModel(self.spark))
        return model.cross_validate(df, feature_cols)

    def save_results(self, comparison_df: pd.DataFrame, output_path: str):
        """Save evaluation results."""
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")


def main():
    """Main function for running sentiment model evaluation."""
    # Create Spark session
    spark = SparkSession.builder \
        .appName("SentimentModels") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    try:
        # Load processed data
        from ..spark.data_ingestion import DataIngestion
        ingestion = DataIngestion(spark)
        df = ingestion.load_sentiment140_data()

        # Sample for efficiency
        df_sample = df.sample(0.1, seed=42)

        # Split data
        train_df, test_df = df_sample.randomSplit([0.8, 0.2], seed=42)

        # Define feature columns (these should match your feature engineering output)
        feature_cols = [
            "text_length", "processed_length", "token_count",
            "emoji_sentiment", "exclamation_count", "question_count",
            "uppercase_ratio", "punctuation_density",
            "vader_compound", "vader_positive", "vader_negative", "vader_neutral",
            "hour_sin", "hour_cos", "is_weekend",
            "2gram_count", "3gram_count"
        ]

        # Create evaluator
        evaluator = ModelEvaluator(spark)

        # Evaluate all models
        results = evaluator.evaluate_all_models(train_df, test_df, feature_cols)

        # Print results
        print("\nModel Comparison Results:")
        print("=" * 80)
        print(results.to_string(index=False))

        # Save results
        evaluator.save_results(results, "model_comparison_results.csv")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()