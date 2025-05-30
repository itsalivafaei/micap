#!/usr/bin/env python3
"""
Test runner for preprocessing module
Provides easy execution and reporting of preprocessing tests
"""

import sys
import os
import unittest
from io import StringIO

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


def run_preprocessing_tests():
    """
    Run all preprocessing tests with detailed reporting
    """
    
    print("="*60)
    print("MICAP Preprocessing Module Test Suite")
    print("="*60)
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover('unit', pattern='test_preprocessing.py')
    
    # Create test runner with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        descriptions=True,
        failfast=False
    )
    
    # Run tests
    print("Running tests...")
    result = runner.run(suite)
    
    # Get the output
    output = stream.getvalue()
    
    # Print results
    print(output)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    # Success rate
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


def run_specific_test_class(class_name):
    """
    Run a specific test class
    
    Args:
        class_name: Name of the test class to run
    """
    
    print(f"Running specific test class: {class_name}")
    
    # Import specific test classes
    from unit.test_preprocessing import (
        TestTextPreprocessor,
        TestPreprocessingUDFs,
        TestTextCleaningFunctions,
        TestEdgeCasesAndErrorHandling,
        TestCreateSparkSession,
        TestIntegrationScenarios
    )
    
    # Map class names to actual classes
    test_classes = {
        'TestTextPreprocessor': TestTextPreprocessor,
        'TestPreprocessingUDFs': TestPreprocessingUDFs,
        'TestTextCleaningFunctions': TestTextCleaningFunctions,
        'TestEdgeCasesAndErrorHandling': TestEdgeCasesAndErrorHandling,
        'TestCreateSparkSession': TestCreateSparkSession,
        'TestIntegrationScenarios': TestIntegrationScenarios
    }
    
    # Get the test class
    test_class = test_classes.get(class_name)
    if not test_class:
        print(f"Test class '{class_name}' not found!")
        print(f"Available test classes: {list(test_classes.keys())}")
        return 1
    
    # Create suite for specific class
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


def main():
    """Main function for test runner"""
    
    if len(sys.argv) > 1:
        # Run specific test class
        class_name = sys.argv[1]
        return run_specific_test_class(class_name)
    else:
        # Run all tests
        return run_preprocessing_tests()


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 