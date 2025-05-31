#!/usr/bin/env python3
"""
Test script for optimized Phase 2 pipeline
Tests the performance improvements and functionality
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from scripts.run_phase2_pipeline import run_phase2_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_optimized_pipeline():
    """Test the optimized Phase 2 pipeline with a very small sample."""
    logger.info("Testing optimized Phase 2 pipeline...")
    
    # Ensure log directory exists
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Test with a very small sample (1% of data)
        start_time = time.time()
        run_phase2_pipeline(sample_size=0.01)
        elapsed = time.time() - start_time
        
        logger.info(f"✓ Pipeline test completed successfully in {elapsed:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_optimized_pipeline()
    if success:
        print("\n✓ Optimized pipeline test PASSED")
        print("You can now run the full pipeline with:")
        print("  python scripts/run_phase2_pipeline.py --sample 0.1")
        print("  python scripts/run_phase2_pipeline.py --full")
    else:
        print("\n✗ Optimized pipeline test FAILED")
        print("Check the logs for details")
        sys.exit(1) 