#!/usr/bin/env python3
"""
Quick verification of key bug fixes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_sentiment_param_grid():
    """Test sentiment models parameter grid"""
    try:
        from tests.unit.test_sentiment_models import TestNaiveBayesModel
        test = TestNaiveBayesModel()
        test.setUp()
        test.test_get_param_grid()
        return True
    except Exception as e:
        print(f"Sentiment param grid error: {e}")
        return False

def test_volume_anomalies():
    """Test volume anomalies detection"""
    try:
        from tests.unit.test_trend_detection import TestAnomalyDetector
        test = TestAnomalyDetector()
        test.setUp()
        test.test_detect_volume_anomalies()
        return True
    except Exception as e:
        print(f"Volume anomalies error: {e}")
        return False

def test_sentiment_anomalies():
    """Test sentiment anomalies detection"""
    try:
        from tests.unit.test_trend_detection import TestAnomalyDetector
        test = TestAnomalyDetector()
        test.setUp()
        test.test_detect_sentiment_anomalies()
        return True
    except Exception as e:
        print(f"Sentiment anomalies error: {e}")
        return False

def main():
    print("Quick Bug Fix Verification")
    print("=" * 40)
    
    tests = [
        ("Sentiment Models - Param Grid", test_sentiment_param_grid),
        ("Trend Detection - Volume Anomalies", test_volume_anomalies),
        ("Trend Detection - Sentiment Anomalies", test_sentiment_anomalies),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"Testing {name}...", end=" ")
        try:
            success = test_func()
            if success:
                print("✅ PASSED")
                results.append(True)
            else:
                print("❌ FAILED")
                results.append(False)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All key bug fixes verified!")
    else:
        print("⚠️ Some issues remain")

if __name__ == "__main__":
    main() 