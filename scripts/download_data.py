"""
Script to download and prepare the Sentiment140 dataset
Handles download, extraction, and initial data validation
"""

import os
import wget
import zipfile
import pandas as pd
from pathlib import Path
import logging


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_sentiment140():
    """
    Download Sentiment140 dataset from Stanford
    The dataset contains 1.6 million labeled tweets
    """

    # Create data directory if it doesn't exist
    data_dir = Path("/Users/ali/Documents/Projects/micap/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Dataset URL
    url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

    # File paths
    zip_path = data_dir / "sentiment140.zip"
    csv_path = data_dir / "training.1600000.processed.noemoticon.csv"

    # Check if already downloaded
    if csv_path.exists():
        logger.info(f"Dataset already exists at {csv_path}")
        return str(csv_path)

    try:
        # Download dataset
        logger.info("Downloading Sentiment140 dataset...")
        wget.download(url, str(zip_path))
        logger.info("\nDownload completed!")

        # Extract zip file
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        logger.info("Extraction completed!")

        # Remove zip file to save space
        os.remove(zip_path)
        logger.info("Cleaned up zip file")

        # Validate dataset
        logger.info("Validating dataset...")
        df = pd.read_csv(csv_path, encoding='latin-1', header=None)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        return str(csv_path)

    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


if __name__ == "__main__":
    download_sentiment140()