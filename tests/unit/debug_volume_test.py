#!/usr/bin/env python3
"""
Debug script for volume anomaly test
"""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def run_volume_test():
    """Run the volume anomaly test with full error output"""
    print("=" * 60)
    print("TESTING VOLUME ANOMALY DETECTION")
    print("=" * 60)
    
    try:
        from tests.unit.test_trend_detection import TestAnomalyDetector
        
        suite = unittest.TestSuite()
        suite.addTest(TestAnomalyDetector('test_detect_volume_anomalies'))
        
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
        result = runner.run(suite)
        
        if result.failures:
            print("\n" + "=" * 40)
            print("FAILURES:")
            print("=" * 40)
            for test, traceback in result.failures:
                print(f"Test: {test}")
                print(f"Traceback:\n{traceback}")
        
        if result.errors:
            print("\n" + "=" * 40)
            print("ERRORS:")
            print("=" * 40)
            for test, traceback in result.errors:
                print(f"Test: {test}")
                print(f"Traceback:\n{traceback}")
                
    except Exception as e:
        print(f"Error importing or running test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_volume_test() 