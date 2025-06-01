"""Topic Analysis Module for MICAP.

Implements topic modeling and sentiment analysis by topic
Uses LDA, clustering, and keyword extraction.
."""

import logging
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import random
from src.utils.path_utils import get_path

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
    """Performs topic modeling and analysis on sentiment data.
    ."""def __init__(self, spark: SparkSession):."""Initialize topic analyzer.

        Args:
            spark: Active SparkSession.
        ."""
        self.spark = spark

        # Extended stop words
        self.stop_words = list(ENGLISH_STOP_WORDS) + [
            'url', 'user', 'rt', 'amp', 'http', 'https',
            'com', 'org', 'www', 'bit', 'ly', 'tweet', 'twitter',
            'say', 'get', 'go', 'know', 'like', 'time', 'people',
            'good', 'bad', 'new', 'day', 'way', 'year', 'think'
        ]

    def extract_topics_lda(self, df: DataFrame,
                           num_topics: int = 10,
                           max_iter: int = 20,
                           vocab_size: int = 1000) -> Tuple[DataFrame, Dict]:
        """Extract topics using Latent Dirichlet Allocation (LDA).

        Args:
            df: Input DataFrame with tokenized text
            num_topics: Number of topics to extract
            max_iter: Maximum iterations for LDA
            vocab_size: Maximum vocabulary size

        Returns:
            Tuple of (DataFrame with topic assignments, topic descriptions).
        ."""
        logger.info(f"Extracting {num_topics} topics using LDA...")

        # Ensure we have the required columns
        if "tokens_lemmatized" not in df.columns:
            if "tokens" in df.columns:
                df = df.withColumn("tokens_lemmatized", col("tokens"))
            else:
                # Create tokens from text if not available
                df = df.withColumn("tokens", split(col("text"), " "))
                df = df.withColumn("tokens_lemmatized", col("tokens"))

        # Remove stop words
        remover = StopWordsRemover(
            inputCol="tokens_lemmatized",
            outputCol="tokens_filtered",
            stopWords=self.stop_words
        )
        df_filtered = remover.transform(df)

        # Filter out empty token lists
        df_filtered = df_filtered.filter(size(col("tokens_filtered")) > 2)

        # Count vectorizer
        cv = CountVectorizer(
            inputCol="tokens_filtered",
            outputCol="raw_features",
            maxDF=0.8,  # Ignore terms that appear in >80% of documents
            minDF=5,    # Ignore terms that appear in <5 documents
            vocabSize=vocab_size
        )

        # IDF
        idf = IDF(
            inputCol="raw_features",
            outputCol="features",
            minDocFreq=5
        )

        # LDA
        lda = LDA(
            k=num_topics,
            maxIter=max_iter,
            seed=42,
            featuresCol="features",
            topicDistributionCol="topic_distribution",
            docConcentration=[1.1],  # Document concentration parameter
            topicConcentration=1.1   # Topic concentration parameter
        )

        # Build pipeline
        pipeline = Pipeline(stages=[cv, idf, lda])

        # Fit pipeline
        logger.info("Training LDA model...")
        model = pipeline.fit(df_filtered)

        # Transform data
        df_topics = model.transform(df_filtered)

        # Extract topic descriptions
        lda_model = model.stages[-1]
        cv_model = model.stages[0]

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
            terms = [(vocab[idx], float(weight)) for idx, weight in
                     zip(term_indices, term_weights)]

            topic_descriptions[topic_id] = {
                'terms': terms,
                'top_words': [term[0] for term in terms[:10]]
            }

        # Add dominant topic to DataFrame
        def get_dominant_topic(distribution):
            if distribution is None or len(distribution) == 0:
                return -1
            return int(np.argmax(distribution))

        dominant_topic_udf = udf(get_dominant_topic, IntegerType())
        df_topics = df_topics.withColumn(
            "dominant_topic",
            dominant_topic_udf(col("topic_distribution"))
        )

        logger.info(f"LDA topic extraction completed with {len(topic_descriptions)} topics")

        return df_topics, topic_descriptions

    def cluster_tweets_by_content(self, df: DataFrame,
                                  num_clusters: int = 20,
                                  use_embeddings: bool = True) -> DataFrame:
        """Cluster tweets based on content similarity.

        Args:
            df: Input DataFrame
            num_clusters: Number of clusters
            use_embeddings: Whether to use word embeddings

        Returns:
            DataFrame with cluster assignments.
        ."""
        logger.info(f"Clustering tweets into {num_clusters} clusters...")

        # Determine features column to use
        if use_embeddings and "word2vec_features" in df.columns:
            features_col = "word2vec_features"
            logger.info("Using Word2Vec embeddings for clustering")
        elif "tfidf_features" in df.columns:
            features_col = "tfidf_features"
            logger.info("Using TF-IDF features for clustering")
        elif "features" in df.columns:
            features_col = "features"
            logger.info("Using available features for clustering")
        else:
            # Create TF-IDF features if none available
            logger.info("Creating TF-IDF features for clustering...")
            
            # Tokenize if needed
            if "tokens_filtered" not in df.columns:
                if "tokens" in df.columns:
                    remover = StopWordsRemover(
                        inputCol="tokens",
                        outputCol="tokens_filtered",
                        stopWords=self.stop_words
                    )
                    df = remover.transform(df)
                else:
                    df = df.withColumn("tokens", split(col("text"), " "))
                    remover = StopWordsRemover(
                        inputCol="tokens",
                        outputCol="tokens_filtered",
                        stopWords=self.stop_words
                    )
                    df = remover.transform(df)
            
            # Create TF-IDF features
            cv = CountVectorizer(
                inputCol="tokens_filtered",
                outputCol="raw_features",
                vocabSize=1000
            )
            idf = IDF(inputCol="raw_features", outputCol="features")
            
            pipeline = Pipeline(stages=[cv, idf])
            model = pipeline.fit(df)
            df = model.transform(df)
            features_col = "features"

        # KMeans clustering
        kmeans = KMeans(
            k=num_clusters,
            featuresCol=features_col,
            predictionCol="cluster",
            maxIter=20,
            seed=42
        )

        # Fit model
        logger.info("Training KMeans model...")
        model = kmeans.fit(df)

        # Transform data
        df_clustered = model.transform(df)

        # Calculate cluster centers and statistics
        centers = model.clusterCenters()
        cluster_stats = df_clustered.groupBy("cluster").agg(
            count("*").alias("cluster_size"),
            avg("sentiment").alias("avg_sentiment")
        ).orderBy("cluster")

        logger.info(f"Created {len(centers)} clusters")
        logger.info("Cluster statistics:")
        cluster_stats.show()

        return df_clustered

    def analyze_sentiment_by_topic(self, df: DataFrame) -> DataFrame:
        """Analyze sentiment distribution by topic.

        Args:
            df: DataFrame with topic assignments and sentiment

        Returns:
            DataFrame with sentiment statistics by topic.
        ."""
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
        """Extract trending topics over time.

        Args:
            df: Input DataFrame
            time_window: Time window for trend analysis
            min_count: Minimum count for trending topic

        Returns:
            DataFrame with trending topics.
        ."""
        logger.info("Extracting trending topics...")

        # Check if we have hashtags or create them from text
        if "hashtags" not in df.columns:
            # Extract hashtags from text
            hashtag_udf = udf(lambda text: [word for word in text.split() if word.startswith('#')], ArrayType(StringType()))
            df = df.withColumn("hashtags", hashtag_udf(col("text")))

        # Check if we have timestamp
        if "timestamp" not in df.columns:
            logger.warning("No timestamp column found, using current time")
            from pyspark.sql.functions import current_timestamp
            df = df.withColumn("timestamp", current_timestamp())

        # Extract hashtags
        hashtag_df = df.select(
            col("timestamp"),
            explode(col("hashtags")).alias("hashtag"),
            col("sentiment")
        ).filter(col("hashtag").isNotNull())

        if hashtag_df.count() == 0:
            logger.warning("No hashtags found in data")
            return self.spark.createDataFrame([], "window struct<start:timestamp,end:timestamp>, hashtag string, count bigint, avg_sentiment double, rank int")

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
                                   min_tweets: int = 50) -> DataFrame:
        """Identify topics with high sentiment polarization.

        Args:
            df: DataFrame with topic assignments
            min_tweets: Minimum tweets per topic

        Returns:
            DataFrame with polarization metrics.
        ."""
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
            col("polarization_score") * when(col("sentiment_stddev").isNull(), 0).otherwise(col("sentiment_stddev"))
        )

        # Rank by controversy
        topic_stats = topic_stats.orderBy(desc("controversy_score"))

        return topic_stats

    def create_topic_network(self, df: DataFrame,
                             topic_descriptions: Dict,
                             min_edge_weight: float = 0.1) -> nx.Graph:
        """Create a network graph of topic relationships.

        Args:
            df: DataFrame with topic assignments
            topic_descriptions: Topic descriptions from LDA
            min_edge_weight: Minimum edge weight to include

        Returns:
            NetworkX graph of topic relationships.
        ."""
        logger.info("Creating topic network...")

        # Calculate topic co-occurrence based on documents with multiple topics
        def get_top_topics(distribution, threshold=0.1):
            if distribution is None or len(distribution) == 0:
                return []
            topics = []
            for i, prob in enumerate(distribution):
                if prob > threshold:
                    topics.append(i)
            return topics[:3]  # Limit to top 3 topics per document

        top_topics_udf = udf(get_top_topics, ArrayType(IntegerType()))

        df_topics = df.withColumn(
            "top_topics",
            top_topics_udf(col("topic_distribution"))
        ).filter(size(col("top_topics")) > 1)  # Only documents with multiple topics

        # Collect topic co-occurrences
        cooccurrences = df_topics.select("top_topics").rdd.flatMap(
            lambda row: [(min(pair), max(pair)) for pair in 
                        [(t1, t2) for i, t1 in enumerate(row.top_topics) 
                         for t2 in row.top_topics[i+1:]]]
        ).countByValue()

        # Create network
        G = nx.Graph()

        # Add nodes with topic information
        for topic_id, desc in topic_descriptions.items():
            G.add_node(topic_id,
                       label=f"Topic {topic_id}",
                       top_words=desc['top_words'][:5])

        # Add edges based on co-occurrence
        total_docs = df_topics.count()
        for (topic1, topic2), count in cooccurrences.items():
            weight = count / total_docs
            if weight >= min_edge_weight:
                G.add_edge(topic1, topic2, weight=weight)

        logger.info(f"Created network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def visualize_topics(self, df: DataFrame,
                         topic_descriptions: Dict,
                         topic_sentiment: DataFrame,
                         output_dir: str = None):
        """Create topic analysis visualizations.

        Args:
            df: DataFrame with topic assignments
            topic_descriptions: Topic descriptions
            topic_sentiment: Sentiment by topic DataFrame
            output_dir: Directory to save visualizations.
        ."""
        if output_dir is None:
            output_dir = str(get_path("data/visualizations/topics"))
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Creating topic visualizations in {output_dir}")

        # Convert to pandas for visualization
        topic_dist = df.groupBy("dominant_topic").count().toPandas()
        sentiment_df = topic_sentiment.toPandas()

        # 1. Topic distribution
        plt.figure(figsize=(12, 6))
        bars = plt.bar(topic_dist['dominant_topic'], topic_dist['count'])
        plt.xlabel('Topic ID')
        plt.ylabel('Number of Tweets')
        plt.title('Topic Distribution')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/topic_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Sentiment by topic
        plt.figure(figsize=(14, 8))
        x = np.arange(len(sentiment_df))
        width = 0.35

        plt.bar(x - width/2, sentiment_df['positive_count'],
               width, label='Positive', color='green', alpha=0.7)
        plt.bar(x + width/2, sentiment_df['negative_count'],
               width, label='Negative', color='red', alpha=0.7)

        plt.xlabel('Topic ID')
        plt.ylabel('Tweet Count')
        plt.title('Sentiment Distribution by Topic')
        plt.xticks(x, sentiment_df['dominant_topic'])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sentiment_by_topic.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Word clouds for top topics
        try:
            for i, (topic_id, desc) in enumerate(list(topic_descriptions.items())[:6]):
                terms = desc.get('terms', [])
                if terms:
                    # Create word frequency dictionary
                    word_freq = {term[0]: float(term[1]) for term in terms[:20]}
                    
                    if word_freq:
                        # Generate word cloud
                        wordcloud = WordCloud(
                            width=800, height=400,
                            background_color='white',
                            max_words=20,
                            colormap='viridis'
                        ).generate_from_frequencies(word_freq)

                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.title(f'Topic {topic_id} - Top Words')
                        plt.tight_layout()
                        plt.savefig(f"{output_dir}/topic_{topic_id}_wordcloud.png",
                                    dpi=300, bbox_inches='tight')
                        plt.close()
        except Exception as e:
            logger.warning(f"Could not create word clouds: {e}")

        # 4. Topic sentiment heatmap
        try:
            if len(sentiment_df) > 0:
                # Prepare data for heatmap
                metrics = ['avg_sentiment', 'positive_ratio']
                if 'avg_vader_compound' in sentiment_df.columns:
                    metrics.append('avg_vader_compound')
                
                heatmap_data = sentiment_df.set_index('dominant_topic')[metrics].T

                plt.figure(figsize=(12, 6))
                sns.heatmap(heatmap_data, annot=True, fmt='.3f',
                           cmap='RdBu_r', center=0.5, cbar_kws={'label': 'Sentiment Score'})
                plt.xlabel('Topic ID')
                plt.ylabel('Metric')
                plt.title('Topic Sentiment Metrics Heatmap')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/topic_sentiment_heatmap.png", dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.warning(f"Could not create sentiment heatmap: {e}")

        # 5. Topic network visualization
        try:
            if len(topic_descriptions) > 1:
                # Create a simple topic similarity network
                plt.figure(figsize=(12, 8))
                
                # Create a simple network based on word overlap
                G = nx.Graph()
                topics = list(topic_descriptions.keys())
                
                # Add nodes
                for topic_id in topics:
                    G.add_node(topic_id)
                
                # Add edges based on word similarity
                for i, topic1 in enumerate(topics):
                    for topic2 in topics[i+1:]:
                        words1 = set(topic_descriptions[topic1]['top_words'][:10])
                        words2 = set(topic_descriptions[topic2]['top_words'][:10])
                        similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                        
                        if similarity > 0.1:  # Threshold for edge creation
                            G.add_edge(topic1, topic2, weight=similarity)
                
                # Draw network
                pos = nx.spring_layout(G, k=3, iterations=50)
                nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                     node_size=1000, alpha=0.7)
                nx.draw_networkx_labels(G, pos)
                
                # Draw edges with varying thickness
                edges = G.edges()
                weights = [G[u][v]['weight'] for u, v in edges]
                nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], alpha=0.6)
                
                plt.title('Topic Similarity Network')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/topic_network.png", dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.warning(f"Could not create topic network: {e}")

        logger.info(f"Topic visualizations saved to {output_dir}")


def main():
    """Demonstrate topic analysis functionality.
    ."""
    from config.spark_config import create_spark_session

    # Create Spark session
    spark = create_spark_session("TopicAnalysis")

    try:
        # Load data
        logger.info("Loading data...")
        data_path = str(get_path("data/processed/pipeline_features"))
        if not os.path.exists(data_path):
            logger.error(f"Data not found at {data_path}. Please run Phase 1 pipeline first.")
            return

        df = spark.read.parquet(data_path)
        logger.info(f"Loaded {df.count()} records")

        # Sample for faster processing
        df_sample = df.sample(0.1, seed=42)
        sample_count = df_sample.count()
        logger.info(f"Using {sample_count} records for analysis")

        # Initialize analyzer
        analyzer = TopicAnalyzer(spark)

        # Extract topics using LDA
        logger.info("\n=== TOPIC EXTRACTION ===")
        df_topics, topic_descriptions = analyzer.extract_topics_lda(
            df_sample, num_topics=8, max_iter=10
        )

        # Analyze sentiment by topic
        logger.info("\n=== SENTIMENT ANALYSIS BY TOPIC ===")
        topic_sentiment = analyzer.analyze_sentiment_by_topic(df_topics)

        # Cluster tweets
        logger.info("\n=== CONTENT CLUSTERING ===")
        df_clustered = analyzer.cluster_tweets_by_content(df_topics, num_clusters=10)

        # Identify polarizing topics
        logger.info("\n=== POLARIZING TOPICS ===")
        polarizing_topics = analyzer.identify_polarizing_topics(df_topics)

        # Extract trending topics
        logger.info("\n=== TRENDING TOPICS ===")
        trending = analyzer.extract_trending_topics(df_sample)

        # Create visualizations
        logger.info("\n=== CREATING VISUALIZATIONS ===")
        analyzer.visualize_topics(df_topics, topic_descriptions, topic_sentiment)

        # Save results
        logger.info("\n=== SAVING RESULTS ===")
        output_path = str(get_path("data/analytics/topics"))
        os.makedirs(output_path, exist_ok=True)

        # Save topic sentiment analysis
        topic_sentiment.coalesce(1).write.mode("overwrite").json(
            f"{output_path}/topic_sentiment"
        )

        # Save topic descriptions
        import json
        with open(f"{output_path}/topic_descriptions.json", 'w') as f:
            json.dump({str(k): v for k, v in topic_descriptions.items()}, f, indent=2)

        # Save polarizing topics
        polarizing_topics.coalesce(1).write.mode("overwrite").json(
            f"{output_path}/polarizing_topics"
        )

        # Display results
        logger.info("\n=== RESULTS SUMMARY ===")
        logger.info("\nTopic descriptions:")
        for topic_id, desc in topic_descriptions.items():
            logger.info(f"Topic {topic_id}: {', '.join(desc['top_words'][:5])}")

        logger.info("\nSentiment by topic:")
        topic_sentiment.select(
            "dominant_topic", "tweet_count", "avg_sentiment",
            "positive_ratio"
        ).show(10, truncate=False)

        logger.info("\nMost polarizing topics:")
        polarizing_topics.select(
            "dominant_topic", "tweet_count", "polarization_score",
            "controversy_score"
        ).show(5, truncate=False)

        if trending.count() > 0:
            logger.info("\nTrending hashtags:")
            trending.show(10, truncate=False)
        else:
            logger.info("No trending hashtags found")

        logger.info(f"\nResults saved to: {output_path}")
        logger.info("✓ Topic analysis completed successfully!")

    except Exception as e:
        logger.error(f"Topic analysis failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop Spark
        spark.stop()


if __name__ == "__main__":
    main()