#!/usr/bin/env python3
"""
Test Runner for Sentiment Models Module
Runs all sentiment models unit tests and provides detailed output
"""

import sys
import os
import unittest
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def run_sentiment_models_tests():
    """Run all sentiment models tests"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    
    # Load specific test module
    suite = loader.loadTestsFromName('test_sentiment_models')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    print("=" * 80)
    print("SENTIMENT MODELS MODULE UNIT TESTS")
    print("=" * 80)
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if result.skipped else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASSED' if success else 'FAILED'}")
    
    return success

if __name__ == '__main__':
    success = run_sentiment_models_tests()
    sys.exit(0 if success else 1) 