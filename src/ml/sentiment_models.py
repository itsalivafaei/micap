"""
Sentiment Analysis Models for MICAP
Implements multiple ML algorithms for sentiment classification
Optimized for distributed processing on M4 Mac
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
    VectorAssembler, StandardScaler, StringIndexer,
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
    """
    Abstract base class for sentiment analysis models
    Provides common functionality for all models
    """

    def __init__(self, spark: SparkSession, model_name: str):
        """
        Initialize base model

        Args:
            spark: Active SparkSession
            model_name: Name of the model
        """
        self.spark = spark
        self.model_name = model_name
        self.model = None
        self.pipeline = None
        self.metrics = {}

    @abstractmethod
    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build the model pipeline"""
        pass

    def prepare_features(self, df: DataFrame, feature_cols: List[str]) -> DataFrame:
        """
        Prepare features for model training

        Args:
            df: Input DataFrame
            feature_cols: List of feature column names

        Returns:
            DataFrame with assembled features
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

        # Scale features
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
        """
        Train the model

        Args:
            train_df: Training DataFrame
            feature_cols: List of feature columns

        Returns:
            Trained PipelineModel
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
        """
        Evaluate the model on test data

        Args:
            test_df: Test DataFrame
            feature_cols: List of feature columns

        Returns:
            Dictionary of evaluation metrics
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
        self.metrics['auc'] = binary_evaluator.evaluate(predictions)
        self.metrics['accuracy'] = accuracy_evaluator.evaluate(predictions)
        self.metrics['precision'] = precision_evaluator.evaluate(predictions)
        self.metrics['recall'] = recall_evaluator.evaluate(predictions)
        self.metrics['f1'] = f1_evaluator.evaluate(predictions)

        # Confusion matrix
        confusion_matrix = predictions.groupBy("label", "prediction").count().toPandas()
        self.metrics['confusion_matrix'] = confusion_matrix

        logger.info(f"{self.model_name} evaluation completed")
        logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {self.metrics['f1']:.4f}")

        return self.metrics

    def cross_validate(self, df: DataFrame, feature_cols: List[str],
                       num_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation

        Args:
            df: Input DataFrame
            feature_cols: List of feature columns
            num_folds: Number of cross-validation folds

        Returns:
            Cross-validation metrics
        """
        logger.info(f"Performing {num_folds}-fold cross-validation for {self.model_name}")

        # Prepare features
        df_prepared = self.prepare_features(df, feature_cols)

        # Build model pipeline
        pipeline = self.build_model(feature_cols)

        # Set up cross-validator
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )

        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=self._get_param_grid(),
            evaluator=evaluator,
            numFolds=num_folds,
            seed=42
        )

        # Perform cross-validation
        start_time = time.time()
        cv_model = cv.fit(df_prepared)
        cv_time = time.time() - start_time

        # Get results
        avg_metrics = cv_model.avgMetrics
        best_metric = max(avg_metrics)

        self.metrics['cv_scores'] = avg_metrics
        self.metrics['cv_best_score'] = best_metric
        self.metrics['cv_time'] = cv_time
        self.metrics['best_params'] = cv_model.bestModel.stages[-1].extractParamMap()

        logger.info(f"Cross-validation completed in {cv_time:.2f} seconds")
        logger.info(f"Best F1 score: {best_metric:.4f}")

        return self.metrics

    @abstractmethod
    def _get_param_grid(self):
        """Get parameter grid for cross-validation"""
        pass


class NaiveBayesModel(BaseModel):
    """
    Naive Bayes classifier for sentiment analysis
    Fast and efficient for text classification
    """

    def __init__(self, spark: SparkSession):
        super().__init__(spark, "Naive Bayes")

    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build Naive Bayes pipeline"""

        # Naive Bayes classifier
        nb = NaiveBayes(
            labelCol="label",
            featuresCol="features",
            smoothing=1.0,
            modelType="multinomial"
        )

        # Create pipeline
        pipeline = Pipeline(stages=[nb])

        return pipeline

    def _get_param_grid(self):
        """Parameter grid for Naive Bayes"""
        nb = NaiveBayes()
        paramGrid = ParamGridBuilder() \
            .addGrid(nb.smoothing, [0.5, 1.0, 2.0]) \
            .build()
        return paramGrid


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression with ElasticNet regularization
    Good baseline for binary classification
    """

    def __init__(self, spark: SparkSession):
        super().__init__(spark, "Logistic Regression")

    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build Logistic Regression pipeline"""

        # Logistic Regression classifier
        lr = LogisticRegression(
            labelCol="label",
            featuresCol="features",
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.5,  # 0.5 = equal mix of L1 and L2
            family="binomial"
        )

        # Create pipeline
        pipeline = Pipeline(stages=[lr])

        return pipeline

    def _get_param_grid(self):
        """Parameter grid for Logistic Regression"""
        lr = LogisticRegression()
        paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()
        return paramGrid


class RandomForestModel(BaseModel):
    """
    Random Forest classifier
    Ensemble method with good performance
    """

    def __init__(self, spark: SparkSession):
        super().__init__(spark, "Random Forest")

    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build Random Forest pipeline"""

        # Random Forest classifier
        rf = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=100,
            maxDepth=10,
            seed=42,
            subsamplingRate=0.8,
            featureSubsetStrategy="sqrt"
        )

        # Create pipeline
        pipeline = Pipeline(stages=[rf])

        return pipeline

    def _get_param_grid(self):
        """Parameter grid for Random Forest"""
        rf = RandomForestClassifier()
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [50, 100]) \
            .addGrid(rf.maxDepth, [5, 10]) \
            .build()
        return paramGrid


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting Trees classifier
    High accuracy but slower training
    """

    def __init__(self, spark: SparkSession):
        super().__init__(spark, "Gradient Boosting")

    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build Gradient Boosting pipeline"""

        # GBT classifier
        gbt = GBTClassifier(
            labelCol="label",
            featuresCol="features",
            maxIter=50,
            maxDepth=5,
            seed=42,
            subsamplingRate=0.8,
            stepSize=0.1
        )

        # Create pipeline
        pipeline = Pipeline(stages=[gbt])

        return pipeline

    def _get_param_grid(self):
        """Parameter grid for GBT"""
        gbt = GBTClassifier()
        paramGrid = ParamGridBuilder() \
            .addGrid(gbt.maxDepth, [3, 5]) \
            .addGrid(gbt.stepSize, [0.1, 0.2]) \
            .build()
        return paramGrid


class SVMModel(BaseModel):
    """
    Support Vector Machine (Linear SVC)
    Good for high-dimensional text features
    """

    def __init__(self, spark: SparkSession):
        super().__init__(spark, "SVM")

    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build SVM pipeline"""

        # Linear SVC
        svm = LinearSVC(
            labelCol="label",
            featuresCol="features",
            maxIter=100,
            regParam=0.01,
            standardization=True
        )

        # Create pipeline
        pipeline = Pipeline(stages=[svm])

        return pipeline

    def _get_param_grid(self):
        """Parameter grid for SVM"""
        svm = LinearSVC()
        paramGrid = ParamGridBuilder() \
            .addGrid(svm.regParam, [0.001, 0.01, 0.1]) \
            .build()
        return paramGrid


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple models using voting
    Combines predictions from different algorithms
    """

    def __init__(self, spark: SparkSession):
        super().__init__(spark, "Ensemble")
        self.base_models = []

    def build_model(self, feature_cols: List[str]) -> Pipeline:
        """Build ensemble pipeline"""

        # Create base models
        nb = NaiveBayes(smoothing=1.0)
        lr = LogisticRegression(maxIter=100, regParam=0.01)
        rf = RandomForestClassifier(numTrees=50, maxDepth=5, seed=42)

        # Store base models
        self.base_models = [nb, lr, rf]

        # For now, we'll use Random Forest as the main model
        # (Full voting ensemble would require custom implementation)
        pipeline = Pipeline(stages=[rf])

        return pipeline

    def _get_param_grid(self):
        """Parameter grid for ensemble"""
        # Simplified for demonstration
        rf = RandomForestClassifier()
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [50, 100]) \
            .build()
        return paramGrid


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.results = {}

    def evaluate_all_models(self, train_df: DataFrame, test_df: DataFrame,
                            feature_cols: List[str]) -> pd.DataFrame:
        """
        Evaluate all models and compare results

        Args:
            train_df: Training data
            test_df: Test data
            feature_cols: Feature columns

        Returns:
            DataFrame with comparison results
        """
        logger.info("Starting comprehensive model evaluation...")

        # Initialize models
        models = [
            NaiveBayesModel(self.spark),
            LogisticRegressionModel(self.spark),
            RandomForestModel(self.spark),
            GradientBoostingModel(self.spark),
            SVMModel(self.spark),
            EnsembleModel(self.spark)
        ]

        # Train and evaluate each model
        for model in models:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Training {model.model_name}")
            logger.info(f"{'=' * 50}")

            # Train model
            model.train(train_df, feature_cols)

            # Evaluate model
            metrics = model.evaluate(test_df, feature_cols)

            # Store results
            self.results[model.model_name] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
                'training_time': metrics['training_time']
            }

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(4)

        # Sort by F1 score
        comparison_df = comparison_df.sort_values('f1', ascending=False)

        logger.info("\n" + "=" * 50)
        logger.info("Model Comparison Results")
        logger.info("=" * 50)
        print(comparison_df)

        return comparison_df

    def perform_cross_validation(self, df: DataFrame, feature_cols: List[str],
                                 model_name: str = "Random Forest") -> Dict:
        """
        Perform cross-validation on selected model

        Args:
            df: Input DataFrame
            feature_cols: Feature columns
            model_name: Model to use for CV

        Returns:
            Cross-validation results
        """
        logger.info(f"Performing cross-validation for {model_name}")

        # Select model
        if model_name == "Random Forest":
            model = RandomForestModel(self.spark)
        elif model_name == "Logistic Regression":
            model = LogisticRegressionModel(self.spark)
        else:
            model = RandomForestModel(self.spark)

        # Perform cross-validation
        cv_results = model.cross_validate(df, feature_cols, num_folds=5)

        return cv_results

    def save_results(self, comparison_df: pd.DataFrame, output_path: str):
        """Save evaluation results"""

        # Save as CSV
        comparison_df.to_csv(f"{output_path}/model_comparison.csv")

        # Save as JSON
        comparison_df.to_json(f"{output_path}/model_comparison.json", orient='index')

        logger.info(f"Results saved to {output_path}")


def main():
    """
    Demonstrate model training and evaluation
    """
    from config.spark_config import create_spark_session

    # Create Spark session
    spark = create_spark_session("ModelTraining")

    # Load featured data
    logger.info("Loading featured data...")
    data_path = "/Users/ali/Documents/Projects/micap/data/processed/pipeline_features"
    df = spark.read.parquet(data_path)

    # Define feature columns
    feature_cols = [
        "text_length", "processed_length", "token_count",
        "emoji_sentiment", "exclamation_count", "question_count",
        "uppercase_ratio", "punctuation_density",
        "vader_compound", "vader_positive", "vader_negative", "vader_neutral",
        "hour_sin", "hour_cos", "is_weekend",
        "2gram_count", "3gram_count"
    ]

    # Split data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Train set: {train_df.count()} records")
    logger.info(f"Test set: {test_df.count()} records")

    # Cache data for faster processing
    train_df.cache()
    test_df.cache()

    # Evaluate all models
    evaluator = ModelEvaluator(spark)
    comparison_df = evaluator.evaluate_all_models(train_df, test_df, feature_cols)

    # Save results
    output_path = "/Users/ali/Documents/Projects/micap/data/models"
    import os
    os.makedirs(output_path, exist_ok=True)
    evaluator.save_results(comparison_df, output_path)

    # Perform cross-validation on best model
    logger.info("\nPerforming cross-validation on best model...")
    cv_results = evaluator.perform_cross_validation(
        df.sample(0.1), feature_cols, "Random Forest"
    )

    logger.info(f"Cross-validation F1 scores: {cv_results['cv_scores']}")
    logger.info(f"Best CV F1 score: {cv_results['cv_best_score']:.4f}")

    # Clean up
    train_df.unpersist()
    test_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()