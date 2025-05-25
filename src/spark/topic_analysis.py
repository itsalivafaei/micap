"""
Topic Analysis Module for MICAP
Implements topic modeling and sentiment analysis by topic
Uses LDA, clustering, and keyword extraction
"""

import logging
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
# import numpy as np
import random

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, udf, explode, split, regexp_replace,
    count, avg, stddev, collect_list, array_distinct,
    concat_ws, size, when, lit, desc, row_number, window, slice, sum as spark_sum
)
from pyspark.sql.types import ArrayType, StringType, DoubleType, IntegerType
from pyspark.sql import Window
from pyspark.ml.feature import (
    CountVectorizer, IDF, StopWordsRemover,
    Word2Vec, BucketedRandomProjectionLSH
)
from pyspark.ml.clustering import LDA, KMeans, BisectingKMeans
from pyspark.ml import Pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class TopicAnalyzer:
    """
    Performs topic modeling and analysis on sentiment data
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize topic analyzer

        Args:
            spark: Active SparkSession
        """
        self.spark = spark

        # Extended stop words
        self.stop_words = list(ENGLISH_STOP_WORDS) + [
            'url', 'user', 'rt', 'amp', 'http', 'https',
            'com', 'org', 'www', 'bit', 'ly'
        ]

    def extract_topics_lda(self, df: DataFrame,
                           num_topics: int = 10,
                           max_iter: int = 20,
                           vocab_size: int = 1000) -> Tuple[DataFrame, Dict]:
        """
        Extract topics using Latent Dirichlet Allocation (LDA)

        Args:
            df: Input DataFrame with tokenized text
            num_topics: Number of topics to extract
            max_iter: Maximum iterations for LDA
            vocab_size: Maximum vocabulary size

        Returns:
            Tuple of (DataFrame with topic assignments, topic descriptions)
        """
        logger.info(f"Extracting {num_topics} topics using LDA...")

        # Prepare text data
        # Remove stop words
        remover = StopWordsRemover(
            inputCol="tokens_lemmatized",
            outputCol="tokens_filtered",
            stopWords=self.stop_words
        )

        # Count vectorizer
        cv = CountVectorizer(
            inputCol="tokens_filtered",
            outputCol="our_raw_features",
            maxDF=0.9,  # Ignore terms that appear in >90% of documents
            minDF=0.01,  # Ignore terms that appear in <10 documents
            vocabSize=vocab_size
        )

        # IDF
        idf = IDF(
            inputCol="our_raw_features",
            outputCol="features",
            minDocFreq=10
        )

        # LDA
        lda = LDA(
            k=num_topics,
            maxIter=max_iter,
            seed=42,
            featuresCol="features",
            topicDistributionCol="topic_distribution"
        )

        # Build pipeline
        pipeline = Pipeline(stages=[remover, cv, idf, lda])

        # Fit pipeline
        model = pipeline.fit(df)

        # Transform data
        df_topics = model.transform(df)

        # Extract topic descriptions
        lda_model = model.stages[-1]
        cv_model = model.stages[1]

        # Get vocabulary
        vocab = cv_model.vocabulary

        # Get topic descriptions
        topics = lda_model.describeTopics(maxTermsPerTopic=20)
        topic_descriptions = {}

        for row in topics.collect():
            topic_id = row['topic']
            term_indices = row['termIndices']
            term_weights = row['termWeights']

            # Get top terms for this topic
            terms = [(vocab[idx], weight) for idx, weight in
                     zip(term_indices, term_weights)]

            topic_descriptions[topic_id] = {
                'terms': terms,
                'top_words': [term[0] for term in terms[:10]]
            }

        # Add dominant topic to DataFrame
        def get_dominant_topic(distribution):
            if distribution is None:
                return -1
            # return int(np.argmax(distribution))
            return max(range(len(distribution)), key=lambda i: distribution[i])

        dominant_topic_udf = udf(get_dominant_topic, IntegerType())
        df_topics = df_topics.withColumn(
            "dominant_topic",
            dominant_topic_udf(col("topic_distribution"))
        )

        logger.info(f"LDA topic extraction completed")

        return df_topics, topic_descriptions

    def cluster_tweets_by_content(self, df: DataFrame,
                                  num_clusters: int = 20,
                                  use_embeddings: bool = True) -> DataFrame:
        """
        Cluster tweets based on content similarity

        Args:
            df: Input DataFrame
            num_clusters: Number of clusters
            use_embeddings: Whether to use word embeddings

        Returns:
            DataFrame with cluster assignments
        """
        logger.info(f"Clustering tweets into {num_clusters} clusters...")

        if use_embeddings and "word2vec_features" in df.columns:
            # Use Word2Vec embeddings
            features_col = "word2vec_features"
        else:
            # Use TF-IDF features
            features_col = "tfidf_features"

        # KMeans clustering
        kmeans = KMeans(
            k=num_clusters,
            featuresCol=features_col,
            predictionCol="cluster",
            maxIter=20,
            seed=42
        )

        # Fit model
        model = kmeans.fit(df)

        # Transform data
        df_clustered = model.transform(df)

        # Calculate cluster centers
        centers = model.clusterCenters()
        logger.info(f"Created {len(centers)} clusters")

        return df_clustered

    def analyze_sentiment_by_topic(self, df: DataFrame) -> DataFrame:
        """
        Analyze sentiment distribution by topic

        Args:
            df: DataFrame with topic assignments and sentiment

        Returns:
            DataFrame with sentiment statistics by topic
        """
        logger.info("Analyzing sentiment by topic...")

        # Group by topic and calculate sentiment statistics
        topic_sentiment = df.groupBy("dominant_topic").agg(
            count("sentiment").alias("tweet_count"),
            avg("sentiment").alias("avg_sentiment"),
            stddev("sentiment").alias("sentiment_stddev"),
            spark_sum(when(col("sentiment") == 1, 1).otherwise(0)).alias("positive_count"),
            spark_sum(when(col("sentiment") == 0, 1).otherwise(0)).alias("negative_count"),
            avg("vader_compound").alias("avg_vader_compound"),
            collect_list("text").alias("sample_tweets")
        )

        # Calculate positive ratio
        topic_sentiment = topic_sentiment.withColumn(
            "positive_ratio",
            col("positive_count") / col("tweet_count")
        )

        # Limit sample tweets
        topic_sentiment = topic_sentiment.withColumn(
            "sample_tweets",
            slice(col("sample_tweets"), 1, 5)
        )

        # Sort by tweet count
        topic_sentiment = topic_sentiment.orderBy(desc("tweet_count"))

        return topic_sentiment

    def extract_trending_topics(self, df: DataFrame,
                                time_window: str = "1 day",
                                min_count: int = 10) -> DataFrame:
        """
        Extract trending topics over time

        Args:
            df: Input DataFrame
            time_window: Time window for trend analysis
            min_count: Minimum count for trending topic

        Returns:
            DataFrame with trending topics
        """
        logger.info("Extracting trending topics...")

        # Extract hashtags
        hashtag_df = df.select(
            col("timestamp"),
            explode(col("hashtags")).alias("hashtag"),
            col("sentiment")
        ).filter(col("hashtag").isNotNull())

        # Aggregate by time window and hashtag
        trending = hashtag_df.groupBy(
            window("timestamp", time_window),
            "hashtag"
        ).agg(
            count("*").alias("count"),
            avg("sentiment").alias("avg_sentiment")
        ).filter(col("count") >= min_count)

        # Rank hashtags within each window
        window_spec = Window.partitionBy("window").orderBy(desc("count"))
        trending = trending.withColumn(
            "rank",
            row_number().over(window_spec)
        )

        # Get top trending per window
        trending = trending.filter(col("rank") <= 10)

        return trending.orderBy("window", "rank")

    def identify_polarizing_topics(self, df: DataFrame,
                                   min_tweets: int = 100) -> DataFrame:
        """
        Identify topics with high sentiment polarization

        Args:
            df: DataFrame with topic assignments
            min_tweets: Minimum tweets per topic

        Returns:
            DataFrame with polarization metrics
        """
        logger.info("Identifying polarizing topics...")

        # Calculate sentiment distribution by topic
        topic_stats = df.groupBy("dominant_topic").agg(
            count("sentiment").alias("tweet_count"),
            avg("sentiment").alias("avg_sentiment"),
            stddev("sentiment").alias("sentiment_stddev"),
            spark_sum(when(col("sentiment") == 1, 1).otherwise(0)).alias("positive_count"),
            spark_sum(when(col("sentiment") == 0, 1).otherwise(0)).alias("negative_count")
        ).filter(col("tweet_count") >= min_tweets)

        # Calculate polarization metrics
        topic_stats = topic_stats.withColumn(
            "positive_ratio",
            col("positive_count") / col("tweet_count")
        ).withColumn(
            "negative_ratio",
            col("negative_count") / col("tweet_count")
        ).withColumn(
            "polarization_score",
            2 * col("positive_ratio") * col("negative_ratio")
        ).withColumn(
            "controversy_score",
            col("polarization_score") * col("sentiment_stddev")
        )

        # Rank by controversy
        topic_stats = topic_stats.orderBy(desc("controversy_score"))

        return topic_stats

    def create_topic_network(self, df: DataFrame,
                             topic_descriptions: Dict,
                             min_edge_weight: float = 0.1) -> nx.Graph:
        """
        Create a network graph of topic relationships

        Args:
            df: DataFrame with topic assignments
            topic_descriptions: Topic descriptions from LDA
            min_edge_weight: Minimum edge weight to include

        Returns:
            NetworkX graph of topic relationships
        """
        logger.info("Creating topic network...")

        # Calculate topic co-occurrence
        # Get tweets with multiple high-probability topics
        def get_top_topics(distribution, threshold=0.2):
            if distribution is None:
                return []
            topics = []
            for i, prob in enumerate(distribution):
                if prob > threshold:
                    topics.append(i)
            return topics

        top_topics_udf = udf(get_top_topics, ArrayType(IntegerType()))

        df_topics = df.withColumn(
            "top_topics",
            top_topics_udf(col("topic_distribution"))
        )

        # Create edges between co-occurring topics
        edges = []

        # This is a simplified version - in practice you'd calculate actual co-occurrence
        num_topics = len(topic_descriptions)
        for i in range(num_topics):
            for j in range(i + 1, num_topics):
                # Simulate edge weight based on topic similarity
                # weight = np.random.random() * 0.5
                weight = random.uniform(0, 0.5)
                if weight > min_edge_weight:
                    edges.append((i, j, weight))

        # Create network
        G = nx.Graph()

        # Add nodes with topic information
        for topic_id, desc in topic_descriptions.items():
            G.add_node(topic_id,
                       label=f"Topic {topic_id}",
                       top_words=desc['top_words'][:5])

        # Add edges
        for i, j, weight in edges:
            G.add_edge(i, j, weight=weight)

        return G

    def visualize_topics(self, df: DataFrame,
                         topic_descriptions: Dict,
                         topic_sentiment: DataFrame,
                         output_dir: str = "/Users/ali/Documents/Projects/micap/data/visualizations"):
        """
        Create topic analysis visualizations

        Args:
            df: DataFrame with topic assignments
            topic_descriptions: Topic descriptions
            topic_sentiment: Sentiment by topic DataFrame
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Topic distribution
        topic_dist = df.groupBy("dominant_topic").count().toPandas()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(topic_dist['dominant_topic'], topic_dist['count'])
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Number of Tweets')
        ax.set_title('Topic Distribution')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_distribution.png", dpi=300)
        plt.close()

        # 2. Sentiment by topic
        sentiment_df = topic_sentiment.toPandas()

        fig, ax = plt.subplots(figsize=(12, 6))
        x = sentiment_df['dominant_topic']
        width = 0.35

        ax.bar(x - width / 2, sentiment_df['positive_count'],
               width, label='Positive', color='green', alpha=0.7)
        ax.bar(x + width / 2, sentiment_df['negative_count'],
               width, label='Negative', color='red', alpha=0.7)

        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Tweet Count')
        ax.set_title('Sentiment Distribution by Topic')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sentiment_by_topic.png", dpi=300)
        plt.close()

        # 3. Word clouds for top topics
        for topic_id in range(min(5, len(topic_descriptions))):
            if topic_id in topic_descriptions:
                words = topic_descriptions[topic_id]['top_words']
                weights = [w[1] for w in topic_descriptions[topic_id]['terms'][:20]]

                # Create word frequency dictionary
                word_freq = {word: weight for word, weight in
                             zip(words[:20], weights)}

                # Generate word cloud
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white'
                ).generate_from_frequencies(word_freq)

                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Topic {topic_id} Word Cloud')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/topic_{topic_id}_wordcloud.png",
                            dpi=300)
                plt.close()

        # 4. Topic sentiment heatmap
        pivot_df = sentiment_df.set_index('dominant_topic')[
            ['avg_sentiment', 'positive_ratio', 'avg_vader_compound']
        ]

        fig, ax = plt.subplots(figsize=(8, 10))
        sns.heatmap(pivot_df.T, annot=True, fmt='.3f',
                    cmap='RdBu_r', center=0.5, ax=ax)
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Metric')
        ax.set_title('Topic Sentiment Metrics')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_sentiment_heatmap.png", dpi=300)
        plt.close()

        logger.info(f"Topic visualizations saved to {output_dir}")


def main():
    """
    Demonstrate topic analysis functionality
    """
    from config.spark_config import create_spark_session

    # Create Spark session
    spark = create_spark_session("TopicAnalysis")

    # After you create 'spark'
    rdd = spark.range(1).rdd.mapPartitions(
        lambda _: [(__import__("numpy").__version__,)]
    )
    df = rdd.toDF(["numpy_version"])
    df.show()

    # Load data
    logger.info("Loading data...")
    df = spark.read.parquet("/Users/ali/Documents/Projects/micap/data/processed/pipeline_features")

    # Sample for faster processing
    df_sample = df.sample(0.1)

    # Initialize analyzer
    analyzer = TopicAnalyzer(spark)

    # Extract topics using LDA
    df_topics, topic_descriptions = analyzer.extract_topics_lda(
        df_sample, num_topics=10
    )

    # Analyze sentiment by topic
    topic_sentiment = analyzer.analyze_sentiment_by_topic(df_topics)

    # Identify polarizing topics
    polarizing_topics = analyzer.identify_polarizing_topics(df_topics)

    # Extract trending topics
    trending = analyzer.extract_trending_topics(df_sample)

    # Create visualizations
    analyzer.visualize_topics(df_topics, topic_descriptions, topic_sentiment)

    # Save results
    output_path = "/Users/ali/Documents/Projects/micap/data/analytics/topics"
    os.makedirs(output_path, exist_ok=True)

    topic_sentiment.coalesce(1).write.mode("overwrite").json(
        f"{output_path}/topic_sentiment"
    )

    # Save topic descriptions
    import json
    with open(f"{output_path}/topic_descriptions.json", 'w') as f:
        json.dump(topic_descriptions, f, indent=2)

    # Show results
    logger.info("\nTopic descriptions:")
    for topic_id, desc in topic_descriptions.items():
        logger.info(f"Topic {topic_id}: {desc['top_words'][:5]}")

    logger.info("\nSentiment by topic:")
    topic_sentiment.select(
        "dominant_topic", "tweet_count", "avg_sentiment",
        "positive_ratio", "avg_vader_compound"
    ).show(10)

    logger.info("\nMost polarizing topics:")
    polarizing_topics.select(
        "dominant_topic", "tweet_count", "polarization_score",
        "controversy_score"
    ).show(5)

    logger.info("\nTrending hashtags:")
    trending.show(20)

    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()