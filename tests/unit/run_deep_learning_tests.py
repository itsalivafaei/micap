#!/usr/bin/env python3
"""
Test Runner for Deep Learning Models
Provides easy execution of deep learning model tests with various options
"""

import sys
import os
import unittest
import argparse
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_specific_test_class(test_class_name, verbose=True):
    """
    Run a specific test class
    
    Args:
        test_class_name: Name of the test class to run
        verbose: Whether to run in verbose mode
    """
    try:
        from test_deep_learning_models import (
            TestDeepLearningModelBase,
            TestLSTMModel,
            TestCNNModel,
            TestTransformerModel,
            TestEvaluateDeepLearningModels,
            TestErrorHandling,
            TestIntegrationScenarios,
            TestModelConfigurationValidation
        )
        
        test_classes = {
            'base': TestDeepLearningModelBase,
            'lstm': TestLSTMModel,
            'cnn': TestCNNModel,
            'transformer': TestTransformerModel,
            'evaluate': TestEvaluateDeepLearningModels,
            'error': TestErrorHandling,
            'integration': TestIntegrationScenarios,
            'config': TestModelConfigurationValidation
        }
        
        if test_class_name not in test_classes:
            print(f"Unknown test class: {test_class_name}")
            print(f"Available classes: {', '.join(test_classes.keys())}")
            return False
        
        test_class = test_classes[test_class_name]
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"Error importing test modules: {e}")
        return False

def run_all_tests(verbose=True):
    """
    Run all deep learning model tests
    
    Args:
        verbose: Whether to run in verbose mode
    """
    try:
        # Import the test module
        from test_deep_learning_models import (
            TestDeepLearningModelBase,
            TestLSTMModel,
            TestCNNModel,
            TestTransformerModel,
            TestEvaluateDeepLearningModels,
            TestErrorHandling,
            TestIntegrationScenarios,
            TestModelConfigurationValidation
        )
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Add all test classes
        test_classes = [
            TestDeepLearningModelBase,
            TestLSTMModel,
            TestCNNModel,
            TestTransformerModel,
            TestEvaluateDeepLearningModels,
            TestErrorHandling,
            TestIntegrationScenarios,
            TestModelConfigurationValidation
        ]
        
        for test_class in test_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
        result = runner.run(test_suite)
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
        
        if result.failures:
            print("\nFAILED TESTS:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"Error importing test modules: {e}")
        print("Make sure you're running from the correct directory and all dependencies are installed.")
        return False

def run_quick_smoke_test():
    """
    Run a quick smoke test to verify basic functionality
    """
    print("Running quick smoke test...")
    try:
        # Test imports
        sys.modules['tensorflow'] = MagicMock()
        sys.modules['tensorflow.keras'] = MagicMock()
        sys.modules['pyspark'] = MagicMock()
        
        from test_deep_learning_models import TestDeepLearningModelBase
        
        # Run one basic test
        suite = unittest.TestSuite()
        suite.addTest(TestDeepLearningModelBase('test_deep_learning_model_initialization'))
        
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("✅ Smoke test passed!")
        else:
            print("❌ Smoke test failed!")
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Smoke test failed with error: {e}")
        return False

def main():
    """Main function to handle command line arguments and run tests"""
    parser = argparse.ArgumentParser(description='Run deep learning model tests')
    parser.add_argument('--class', '-c', dest='test_class', 
                       help='Run specific test class (base, lstm, cnn, transformer, evaluate, error, integration, config)')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Run tests in quiet mode')
    parser.add_argument('--smoke', '-s', action='store_true',
                       help='Run quick smoke test only')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available test classes')
    
    args = parser.parse_args()
    
    # List available test classes
    if args.list:
        print("Available test classes:")
        print("  base         - TestDeepLearningModelBase")
        print("  lstm         - TestLSTMModel") 
        print("  cnn          - TestCNNModel")
        print("  transformer  - TestTransformerModel")
        print("  evaluate     - TestEvaluateDeepLearningModels")
        print("  error        - TestErrorHandling")
        print("  integration  - TestIntegrationScenarios")
        print("  config       - TestModelConfigurationValidation")
        return
    
    # Run smoke test
    if args.smoke:
        success = run_quick_smoke_test()
        sys.exit(0 if success else 1)
    
    # Run specific test class
    if args.test_class:
        success = run_specific_test_class(args.test_class, verbose=not args.quiet)
        sys.exit(0 if success else 1)
    
    # Run all tests (default)
    success = run_all_tests(verbose=not args.quiet)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 