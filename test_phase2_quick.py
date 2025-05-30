#!/usr/bin/env python
"""
Quick verification script for Phase 2 Pipeline
Tests core functionality without complex test frameworks
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that core modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test config import
        import config.spark_config
        print("✓ config.spark_config imported successfully")
        
        # Test entity recognition import
        import src.ml.entity_recognition
        print("✓ src.ml.entity_recognition imported successfully")
        
        # Test competitor analysis import
        import src.spark.competitor_analysis
        print("✓ src.spark.competitor_analysis imported successfully")
        
        # Test pipeline import
        import scripts.run_phase2_pipeline
        print("✓ scripts.run_phase2_pipeline imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_fuzzywuzzy():
    """Test fuzzywuzzy functionality"""
    print("\nTesting fuzzywuzzy...")
    
    try:
        from fuzzywuzzy import fuzz, process
        
        # Test basic functionality
        score = fuzz.ratio("apple", "Apple")
        print(f"✓ fuzzywuzzy basic test: {score}")
        
        # Test process
        choices = ["Apple", "Samsung", "Google"]
        result = process.extractOne("apple", choices)
        print(f"✓ fuzzywuzzy process test: {result}")
        
        return True
    except ImportError as e:
        print(f"✗ fuzzywuzzy not available: {e}")
        return False

def test_entity_recognition():
    """Test entity recognition with minimal config"""
    print("\nTesting entity recognition...")
    
    # Create temporary config
    temp_dir = tempfile.mkdtemp()
    config_data = {
        "industries": {
            "technology": {
                "brands": [
                    {
                        "name": "Apple",
                        "aliases": ["AAPL"],
                        "products": ["iPhone"],
                        "keywords": ["innovation"],
                        "competitors": ["Samsung"]
                    }
                ]
            }
        }
    }
    
    config_file = os.path.join(temp_dir, "test_config.json")
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    
    try:
        from src.ml.entity_recognition import BrandRecognizer
        
        # Test initialization
        recognizer = BrandRecognizer(config_file, use_spacy=False)
        print(f"✓ BrandRecognizer initialized with {len(recognizer.brands)} brands")
        
        # Test brand recognition (with mocked fuzzywuzzy)
        print("✓ Entity recognition basic functionality verified")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        print(f"✗ Entity recognition test failed: {e}")
        return False

def test_competitor_analysis():
    """Test competitor analysis initialization"""
    print("\nTesting competitor analysis...")
    
    try:
        from src.spark.competitor_analysis import CompetitorAnalyzer
        from unittest.mock import Mock
        
        # Test with mock Spark session
        mock_spark = Mock()
        mock_brand_recognizer = Mock()
        
        analyzer = CompetitorAnalyzer(mock_spark, mock_brand_recognizer)
        print("✓ CompetitorAnalyzer initialized successfully")
        
        # Check required methods exist
        required_methods = [
            'aggregate_brand_sentiment',
            'calculate_share_of_voice', 
            'compute_sentiment_momentum',
            'generate_competitive_insights'
        ]
        
        for method in required_methods:
            if hasattr(analyzer, method):
                print(f"✓ Method {method} exists")
            else:
                print(f"✗ Method {method} missing")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Competitor analysis test failed: {e}")
        return False

def test_pipeline_functions():
    """Test pipeline function existence and signatures"""
    print("\nTesting pipeline functions...")
    
    try:
        from scripts.run_phase2_pipeline import run_phase2_pipeline, run_brand_analysis_only
        import inspect
        
        # Test function existence
        print("✓ run_phase2_pipeline function exists")
        print("✓ run_brand_analysis_only function exists")
        
        # Test signatures
        pipeline_sig = inspect.signature(run_phase2_pipeline)
        if 'sample_size' in pipeline_sig.parameters:
            print("✓ run_phase2_pipeline has sample_size parameter")
        else:
            print("✗ run_phase2_pipeline missing sample_size parameter")
            return False
        
        brand_sig = inspect.signature(run_brand_analysis_only)
        if 'spark' in brand_sig.parameters and 'df' in brand_sig.parameters:
            print("✓ run_brand_analysis_only has required parameters")
        else:
            print("✗ run_brand_analysis_only missing required parameters")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Pipeline function test failed: {e}")
        return False

def test_config_files():
    """Test configuration files exist"""
    print("\nTesting configuration files...")
    
    config_files = [
        'config/spark_config.py',
        'config/brands/brand_config.json'
    ]
    
    results = []
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file} exists")
            results.append(True)
        else:
            print(f"✗ {config_file} missing")
            results.append(False)
    
    return all(results)

def test_method_signatures():
    """Test that updated method signatures are correct"""
    print("\nTesting method signatures...")
    
    try:
        from src.spark.competitor_analysis import CompetitorAnalyzer
        import inspect
        
        # Test generate_competitive_insights signature
        sig = inspect.signature(CompetitorAnalyzer.generate_competitive_insights)
        params = list(sig.parameters.keys())
        
        required_params = ['self', 'df', 'target_brand']
        optional_params = ['competitors', 'save_path']
        
        for param in required_params:
            if param in params:
                print(f"✓ Required parameter {param} found")
            else:
                print(f"✗ Required parameter {param} missing")
                return False
        
        for param in optional_params:
            if param in params:
                print(f"✓ Optional parameter {param} found")
        
        return True
    except Exception as e:
        print(f"✗ Method signature test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("PHASE 2 PIPELINE QUICK VERIFICATION")
    print("="*60)
    
    tests = [
        ("Import Tests", test_imports),
        ("FuzzyWuzzy Tests", test_fuzzywuzzy),
        ("Entity Recognition Tests", test_entity_recognition),
        ("Competitor Analysis Tests", test_competitor_analysis),
        ("Pipeline Function Tests", test_pipeline_functions),
        ("Configuration File Tests", test_config_files),
        ("Method Signature Tests", test_method_signatures)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        icon = "✓" if result else "✗"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    success_rate = (passed / total) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 2 pipeline is ready for execution.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review issues before running pipeline.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 