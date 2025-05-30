#!/usr/bin/env python3
"""
Comprehensive verification script for all bug fixes
"""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def run_specific_tests():
    """Run specific tests that were previously failing"""
    print("=" * 80)
    print("VERIFYING BUG FIXES - RUNNING PREVIOUSLY FAILING TESTS")
    print("=" * 80)
    
    test_results = {}
    
    # Test 1: Sentiment Models - Parameter Grid
    print("\n1. Testing Sentiment Models - Parameter Grid...")
    try:
        from tests.unit.test_sentiment_models import TestNaiveBayesModel
        
        suite = unittest.TestSuite()
        suite.addTest(TestNaiveBayesModel('test_get_param_grid'))
        
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        if result.wasSuccessful():
            test_results['sentiment_param_grid'] = "✅ PASSED"
        else:
            test_results['sentiment_param_grid'] = "❌ FAILED"
            
    except Exception as e:
        test_results['sentiment_param_grid'] = f"❌ ERROR: {e}"
    
    # Test 2: Trend Detection - Volume Anomalies
    print("2. Testing Trend Detection - Volume Anomalies...")
    try:
        from tests.unit.test_trend_detection import TestAnomalyDetector
        
        suite = unittest.TestSuite()
        suite.addTest(TestAnomalyDetector('test_detect_volume_anomalies'))
        
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        if result.wasSuccessful():
            test_results['trend_volume_anomalies'] = "✅ PASSED"
        else:
            test_results['trend_volume_anomalies'] = "❌ FAILED"
            
    except Exception as e:
        test_results['trend_volume_anomalies'] = f"❌ ERROR: {e}"
    
    # Test 3: Trend Detection - Sentiment Anomalies
    print("3. Testing Trend Detection - Sentiment Anomalies...")
    try:
        from tests.unit.test_trend_detection import TestAnomalyDetector
        
        suite = unittest.TestSuite()
        suite.addTest(TestAnomalyDetector('test_detect_sentiment_anomalies'))
        
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        if result.wasSuccessful():
            test_results['trend_sentiment_anomalies'] = "✅ PASSED"
        else:
            test_results['trend_sentiment_anomalies'] = "❌ FAILED"
            
    except Exception as e:
        test_results['trend_sentiment_anomalies'] = f"❌ ERROR: {e}"
    
    # Test 4: Trend Detection - Emerging Topics
    print("4. Testing Trend Detection - Emerging Topics...")
    try:
        from tests.unit.test_trend_detection import TestTopicModeler
        
        suite = unittest.TestSuite()
        suite.addTest(TestTopicModeler('test_detect_emerging_topics'))
        
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        if result.wasSuccessful():
            test_results['trend_emerging_topics'] = "✅ PASSED"
        else:
            test_results['trend_emerging_topics'] = "❌ FAILED"
            
    except Exception as e:
        test_results['trend_emerging_topics'] = f"❌ ERROR: {e}"
    
    # Test 5: Trend Detection - Forecast Sentiment Trends
    print("5. Testing Trend Detection - Forecast Sentiment Trends...")
    try:
        from tests.unit.test_trend_detection import TestTrendForecaster
        
        suite = unittest.TestSuite()
        suite.addTest(TestTrendForecaster('test_forecast_sentiment_trends'))
        
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        if result.wasSuccessful():
            test_results['trend_forecast_sentiment'] = "✅ PASSED"
        else:
            test_results['trend_forecast_sentiment'] = "❌ FAILED"
            
    except Exception as e:
        test_results['trend_forecast_sentiment'] = f"❌ ERROR: {e}"
    
    # Test 6: Trend Detection - Forecast Topic Trends
    print("6. Testing Trend Detection - Forecast Topic Trends...")
    try:
        from tests.unit.test_trend_detection import TestTrendForecaster
        
        suite = unittest.TestSuite()
        suite.addTest(TestTrendForecaster('test_forecast_topic_trends'))
        
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        if result.wasSuccessful():
            test_results['trend_forecast_topics'] = "✅ PASSED"
        else:
            test_results['trend_forecast_topics'] = "❌ FAILED"
            
    except Exception as e:
        test_results['trend_forecast_topics'] = f"❌ ERROR: {e}"
    
    # Test 7: Trend Detection - Viral Potential
    print("7. Testing Trend Detection - Viral Potential...")
    try:
        from tests.unit.test_trend_detection import TestViralityPredictor
        
        suite = unittest.TestSuite()
        suite.addTest(TestViralityPredictor('test_identify_viral_potential_simple'))
        
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        if result.wasSuccessful():
            test_results['trend_viral_potential'] = "✅ PASSED"
        else:
            test_results['trend_viral_potential'] = "❌ FAILED"
            
    except Exception as e:
        test_results['trend_viral_potential'] = f"❌ ERROR: {e}"
    
    # Test 8: Trend Detection - Anomaly with No Data
    print("8. Testing Trend Detection - Anomaly with No Data...")
    try:
        from tests.unit.test_trend_detection import TestErrorHandling
        
        suite = unittest.TestSuite()
        suite.addTest(TestErrorHandling('test_anomaly_detection_with_no_data'))
        
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        if result.wasSuccessful():
            test_results['trend_anomaly_no_data'] = "✅ PASSED"
        else:
            test_results['trend_anomaly_no_data'] = "❌ FAILED"
            
    except Exception as e:
        test_results['trend_anomaly_no_data'] = f"❌ ERROR: {e}"
    
    # Print Results Summary
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS SUMMARY")
    print("=" * 80)
    
    for test_name, result in test_results.items():
        print(f"{test_name:30}: {result}")
    
    # Overall status
    passed_tests = sum(1 for result in test_results.values() if "✅ PASSED" in result)
    total_tests = len(test_results)
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL BUGS FIXED! All previously failing tests are now passing.")
        return True
    else:
        print("⚠️  Some issues remain. See details above.")
        return False

if __name__ == "__main__":
    success = run_specific_tests()
    sys.exit(0 if success else 1) 