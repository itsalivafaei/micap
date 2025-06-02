"""Deep Learning Models for Sentiment Analysis.

Implements LSTM and CNN models using TensorFlow.

Optimized for M4 Mac with Metal acceleration.
"""

import os
import numpy as np
import pandas as pd
# from pandas import DataFrame, concat
import logging
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, collect_list

# Configure TensorFlow for M4 Mac
# tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def spark_to_pandas_stream(df, batch=20_000):
#     parts = (DataFrame(rows, columns=df.columns)
#              for rows in df.toLocalIterator(batch))
#     return concat(parts, ignore_index=True)

class DeepLearningModel:
    """Base class for deep learning models."""
    def __init__(self, spark: SparkSession, max_words: int = 10000,
                 max_length: int = 100):
        """Initialize deep learning model.

        Args:
            spark: SparkSession instance.
            max_words: Maximum vocabulary size.
            max_length: Maximum sequence length.
        """
        self.spark = spark
        self.max_words = max_words
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None
        self.history = None

    def spark_to_pandas_stream(self, df, batch_size=20_000):
        """Stream Spark DataFrame to Pandas in batches to avoid memory issues.
        
        This prevents macOS crashes and optimizes RAM usage.

        Args:
            df: Spark DataFrame to convert.
            batch_size: Number of rows to process at a time.

        Returns:
            Pandas DataFrame.
        """
        # Use toLocalIterator() which returns one row at a time.
        # Then batch them manually for efficiency
        iterator = df.toLocalIterator()

        chunks = []
        current_batch = []

        for row in iterator:
            current_batch.append(row)

            if len(current_batch) >= batch_size:
                # Convert batch to DataFrame
                chunk_df = pd.DataFrame(current_batch, columns=df.columns)
                chunks.append(chunk_df)
                current_batch = []

        # Don't forget the last batch
        if current_batch:
            chunk_df = pd.DataFrame(current_batch, columns=df.columns)
            chunks.append(chunk_df)

        # Concatenate all chunks
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.DataFrame(columns=df.columns)

    def prepare_text_data(self, df: DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare text data for deep learning.

        Args:
            df: Spark DataFrame with text and sentiment.

        Returns:
            Tuple of (X, y) arrays.
        """
        logger.info("Preparing text data for deep learning...")

        # Convert to Pandas for text processing
        # pdf = df.select("text", "sentiment").toPandas()
        ### Steam one Arrow batch at a time
        df_small = (df.sample(0.30, seed=42)  # pull ≤50 k rows
                    .select("text", "sentiment")
                    .repartition(8))  # few, large partitions
        pdf = self.spark_to_pandas_stream(df_small)

        # Fit tokenizer on texts
        self.tokenizer.fit_on_texts(pdf['text'].values)

        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(pdf['text'].values)

        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_length)
        y = pdf['sentiment'].values

        logger.info(f"Prepared {len(X)} samples")
        logger.info(f"Vocabulary size: {len(self.tokenizer.word_index)}")

        return X, y


class LSTMModel(DeepLearningModel):
    """LSTM model for sentiment analysis."""
    
    def build_model(self, embedding_dim: int = 128) -> models.Model:
        """Build LSTM model architecture.

        Args:
            embedding_dim: Dimension of word embeddings.

        Returns:
            Compiled Keras model.
        """
        logger.info("Building LSTM model...")

        model = models.Sequential([
            layers.Embedding(self.max_words, embedding_dim,
                           input_length=self.max_length),
            layers.SpatialDropout1D(0.2),
            layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        # Compile with M4 optimized settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        self.model = model
        return model

    def train(self, X: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2,
              epochs: int = 10, batch_size: int = 64) -> Dict:
        """Train LSTM model.

        Args:
            X: Input sequences.
            y: Labels.
            validation_split: Validation split ratio.
            epochs: Number of epochs.
            batch_size: Batch size.

        Returns:
            Training history.
        """
        logger.info("Training LSTM model...")

        # Define callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001
        )

        # Train model
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        logger.info("LSTM training completed")
        return self.history.history


class CNNModel(DeepLearningModel):
    """CNN model for sentiment analysis."""
    
    def build_model(self, embedding_dim: int = 128) -> models.Model:
        """Build CNN model architecture.

        Args:
            embedding_dim: Dimension of word embeddings.

        Returns:
            Compiled Keras model.
        """
        logger.info("Building CNN model...")

        model = models.Sequential([
            layers.Embedding(self.max_words, embedding_dim,
                           input_length=self.max_length),
            layers.SpatialDropout1D(0.2),
            layers.Conv1D(64, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        # Compile with M4 optimized settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        self.model = model
        return model

    def train(self, X: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2,
              epochs: int = 10, batch_size: int = 32) -> Dict:
        """Train CNN model.

        Args:
            X: Input sequences.
            y: Labels.
            validation_split: Validation split ratio.
            epochs: Number of epochs.
            batch_size: Batch size.

        Returns:
            Training history.
        """
        logger.info("Training CNN model...")

        # Define callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001
        )

        # Train model
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        logger.info("CNN training completed")
        return self.history.history


class TransformerModel(DeepLearningModel):
    """Transformer model for sentiment analysis.
    
    Implements a simplified transformer architecture.
    """

    def build_model(self, embedding_dim: int = 128,
                   num_heads: int = 4) -> models.Model:
        """Build Transformer model architecture.

        Args:
            embedding_dim: Dimension of word embeddings.
            num_heads: Number of attention heads.

        Returns:
            Compiled Keras model.
        """
        logger.info("Building Transformer model...")

        # Input layer
        inputs = layers.Input(shape=(self.max_length,))

        # Embedding layer
        embedding = layers.Embedding(self.max_words, embedding_dim)(inputs)

        # Positional encoding (simplified)
        positions = tf.range(start=0, limit=self.max_length, delta=1)
        pos_embedding = layers.Embedding(self.max_length, embedding_dim)(positions)
        x = embedding + pos_embedding

        # Multi-head attention
        attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim
        )(x, x)

        # Add & norm
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)

        # Feed forward
        ff = layers.Dense(embedding_dim * 2, activation='relu')(x)
        ff = layers.Dense(embedding_dim)(ff)

        # Add & norm
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)

        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Classification head
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs, outputs)

        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        self.model = model
        return model

    def train(self, X: np.ndarray, y: np.ndarray,
              validation_split: float = 0.2,
              epochs: int = 10, batch_size: int = 32) -> Dict:
        """Train Transformer model.

        Args:
            X: Input sequences.
            y: Labels.
            validation_split: Validation split ratio.
            epochs: Number of epochs.
            batch_size: Batch size.

        Returns:
            Training history.
        """
        logger.info("Training Transformer model...")

        # Define callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001
        )

        # Train model
        self.history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        logger.info("Transformer training completed")
        return self.history.history


def evaluate_deep_learning_models(spark: SparkSession,
                                sample_size: float = 0.1) -> Dict:
    """Evaluate all deep learning models.

    Args:
        spark: SparkSession instance.
        sample_size: Fraction of data to use for evaluation.

    Returns:
        Dictionary with evaluation results.
    """
    logger.info("Starting deep learning model evaluation...")

    # Load sample data
    from ..spark.data_ingestion import DataIngestion
    ingestion = DataIngestion(spark)
    df = ingestion.load_sentiment140_data()

    # Sample for efficiency
    df_sample = df.sample(sample_size, seed=42)

    results = {}

    # Define models to test
    models = {
        'LSTM': LSTMModel,
        'CNN': CNNModel,
        'Transformer': TransformerModel
    }

    for model_name, model_class in models.items():
        try:
            logger.info(f"Evaluating {model_name} model...")

            # Initialize model
            model = model_class(spark)

            # Prepare data
            X, y = model.prepare_text_data(df_sample)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Build and train model
            model.build_model()
            history = model.train(X_train, y_train, epochs=5)

            # Evaluate
            test_loss, test_acc, test_auc = model.model.evaluate(
                X_test, y_test, verbose=0
            )

            results[model_name] = {
                'test_accuracy': float(test_acc),
                'test_auc': float(test_auc),
                'test_loss': float(test_loss),
                'training_history': history
            }

            logger.info(f"{model_name} - Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")

        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            results[model_name] = {'error': str(e)}

    logger.info("Deep learning evaluation completed")
    return results


if __name__ == "__main__":
    # Create Spark session
    spark = SparkSession.builder \
        .appName("DeepLearningModels") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    try:
        # Run evaluation
        results = evaluate_deep_learning_models(spark, sample_size=0.05)

        # Print results
        for model_name, metrics in results.items():
            print(f"\n{model_name} Results:")
            if 'error' in metrics:
                print(f"  Error: {metrics['error']}")
            else:
                print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
                print(f"  AUC: {metrics['test_auc']:.4f}")
                print(f"  Loss: {metrics['test_loss']:.4f}")

    finally:
        spark.stop()