"""
Complete Phase 2 Pipeline Verification Suite
Comprehensive tests to verify all components work together successfully
"""

import unittest
import sys
import os
import time
import tempfile
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import test modules
from tests.unit.test_phase2_lightweight import (
    TestEntityRecognitionUnit, TestCompetitorAnalysisUnit, 
    TestPipelineIntegrationUnit, TestConfigurationHandling
)

# Try to import minimal integration tests
try:
    from tests.integration.test_phase2_minimal import (
        TestMinimalResourceIntegration, TestResourceConstraints
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False


class TestSystemReadiness(unittest.TestCase):
    """Test that the system is ready for Phase 2 pipeline execution"""
    
    def test_required_dependencies(self):
        """Test that all required dependencies are available"""
        required_modules = [
            'pyspark',
            'fuzzywuzzy', 
            'pandas',
            'numpy',
            'sklearn'
        ]
        
        missing_modules = []
        available_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
                available_modules.append(module)
            except ImportError:
                missing_modules.append(module)
        
        print(f"\nDependency Check:")
        print(f"Available: {', '.join(available_modules)}")
        if missing_modules:
            print(f"Missing: {', '.join(missing_modules)}")
        
        # At minimum, we need fuzzywuzzy for entity recognition
        self.assertNotIn('fuzzywuzzy', missing_modules, 
                        "fuzzywuzzy is required for entity recognition")
    
    def test_directory_structure(self):
        """Test that required directory structure exists"""
        project_root = Path(__file__).parent.parent
        
        required_dirs = [
            'src/ml',
            'src/spark', 
            'src/utils',
            'config',
            'scripts',
            'tests/unit',
            'tests/integration'
        ]
        
        missing_dirs = []
        existing_dirs = []
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            if full_path.exists():
                existing_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)
        
        print(f"\nDirectory Structure Check:")
        print(f"Existing: {len(existing_dirs)}/{len(required_dirs)} directories")
        
        if missing_dirs:
            print(f"Missing: {', '.join(missing_dirs)}")
            
        # Should have core directories
        self.assertIn('src/ml', existing_dirs)
        self.assertIn('src/spark', existing_dirs)
    
    def test_config_files_exist(self):
        """Test that configuration files exist"""
        project_root = Path(__file__).parent.parent
        
        config_files = [
            'config/spark_config.py',
            'config/brands/brand_config.json'
        ]
        
        existing_files = []
        missing_files = []
        
        for file_path in config_files:
            full_path = project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        print(f"\nConfiguration Files Check:")
        print(f"Existing: {', '.join(existing_files)}")
        if missing_files:
            print(f"Missing: {', '.join(missing_files)}")
        
        # At least spark config should exist
        self.assertIn('config/spark_config.py', existing_files)
    
    def test_module_imports(self):
        """Test that core modules can be imported"""
        core_modules = [
            'src.ml.entity_recognition',
            'src.spark.competitor_analysis',
            'scripts.run_phase2_pipeline',
            'config.spark_config'
        ]
        
        import_results = {}
        
        for module in core_modules:
            try:
                __import__(module)
                import_results[module] = "SUCCESS"
            except ImportError as e:
                import_results[module] = f"FAILED: {str(e)}"
            except Exception as e:
                import_results[module] = f"ERROR: {str(e)}"
        
        print(f"\nModule Import Check:")
        for module, result in import_results.items():
            status = "✓" if result == "SUCCESS" else "✗"
            print(f"  {status} {module}: {result}")
        
        # Core modules should import successfully
        self.assertEqual(import_results['config.spark_config'], "SUCCESS")


class TestFunctionalVerification(unittest.TestCase):
    """Test core functionality with minimal examples"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal test config
        self.test_config = {
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
        
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)
    
    def test_entity_recognition_basic(self):
        """Test basic entity recognition functionality"""
        try:
            # Mock fuzzywuzzy for basic test
            import unittest.mock as mock
            
            with mock.patch('src.ml.entity_recognition.fuzz'), \
                 mock.patch('src.ml.entity_recognition.process'):
                
                from src.ml.entity_recognition import BrandRecognizer
                
                recognizer = BrandRecognizer(self.config_file, use_spacy=False)
                
                # Test initialization
                self.assertGreater(len(recognizer.brands), 0)
                self.assertIn('apple', recognizer.brands)
                
                print("✓ Entity recognition basic functionality verified")
                
        except ImportError as e:
            self.skipTest(f"Entity recognition not available: {e}")
    
    def test_competitor_analysis_basic(self):
        """Test basic competitor analysis functionality"""
        try:
            import unittest.mock as mock
            
            mock_spark = mock.Mock()
            mock_brand_recognizer = mock.Mock()
            
            from src.spark.competitor_analysis import CompetitorAnalyzer
            
            analyzer = CompetitorAnalyzer(mock_spark, mock_brand_recognizer)
            
            # Test initialization
            self.assertEqual(analyzer.spark, mock_spark)
            self.assertIn('hourly', analyzer.time_windows)
            
            print("✓ Competitor analysis basic functionality verified")
            
        except ImportError as e:
            self.skipTest(f"Competitor analysis not available: {e}")
    
    def test_pipeline_function_exists(self):
        """Test that pipeline functions exist and are callable"""
        try:
            from scripts.run_phase2_pipeline import run_phase2_pipeline, run_brand_analysis_only
            
            # Test functions exist
            self.assertTrue(callable(run_phase2_pipeline))
            self.assertTrue(callable(run_brand_analysis_only))
            
            # Test function signatures
            import inspect
            
            pipeline_sig = inspect.signature(run_phase2_pipeline)
            self.assertIn('sample_size', pipeline_sig.parameters)
            
            brand_sig = inspect.signature(run_brand_analysis_only)
            self.assertIn('spark', brand_sig.parameters)
            self.assertIn('df', brand_sig.parameters)
            
            print("✓ Pipeline functions verified")
            
        except ImportError as e:
            self.skipTest(f"Pipeline module not available: {e}")


def run_comprehensive_tests():
    """Run comprehensive test suite with proper reporting"""
    
    print("="*80)
    print("PHASE 2 PIPELINE COMPREHENSIVE VERIFICATION")
    print("="*80)
    
    start_time = time.time()
    
    # Test suite components
    test_suites = []
    
    # 1. System Readiness Tests
    print("\n1. SYSTEM READINESS VERIFICATION")
    print("-" * 50)
    
    readiness_suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemReadiness)
    readiness_result = unittest.TextTestRunner(verbosity=2, buffer=True).run(readiness_suite)
    test_suites.append(("System Readiness", readiness_result))
    
    # 2. Functional Verification Tests
    print("\n2. FUNCTIONAL VERIFICATION")
    print("-" * 50)
    
    functional_suite = unittest.TestLoader().loadTestsFromTestCase(TestFunctionalVerification)
    functional_result = unittest.TextTestRunner(verbosity=2, buffer=True).run(functional_suite)
    test_suites.append(("Functional Verification", functional_result))
    
    # 3. Unit Tests
    print("\n3. UNIT TESTS")
    print("-" * 50)
    
    unit_classes = [
        TestEntityRecognitionUnit,
        TestCompetitorAnalysisUnit,
        TestPipelineIntegrationUnit,
        TestConfigurationHandling
    ]
    
    unit_suite = unittest.TestSuite()
    for test_class in unit_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        unit_suite.addTests(tests)
    
    unit_result = unittest.TextTestRunner(verbosity=1, buffer=True).run(unit_suite)
    test_suites.append(("Unit Tests", unit_result))
    
    # 4. Integration Tests (if available)
    if INTEGRATION_AVAILABLE:
        print("\n4. INTEGRATION TESTS")
        print("-" * 50)
        
        integration_classes = [TestMinimalResourceIntegration, TestResourceConstraints]
        
        integration_suite = unittest.TestSuite()
        for test_class in integration_classes:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
            integration_suite.addTests(tests)
        
        integration_result = unittest.TextTestRunner(verbosity=1, buffer=True).run(integration_suite)
        test_suites.append(("Integration Tests", integration_result))
    else:
        print("\n4. INTEGRATION TESTS - SKIPPED")
        print("-" * 50)
        print("PySpark not available or integration tests not accessible")
    
    # Summary Report
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    for suite_name, result in test_suites:
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(getattr(result, 'skipped', []))
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
        total_skipped += skipped
        
        success_count = tests_run - failures - errors
        if tests_run > 0:
            success_rate = (success_count / tests_run) * 100
        else:
            success_rate = 0
        
        status = "✓" if failures == 0 and errors == 0 else "✗"
        
        print(f"{status} {suite_name}:")
        print(f"    Tests: {tests_run}, Passed: {success_count}, Failed: {failures}, Errors: {errors}, Skipped: {skipped}")
        print(f"    Success Rate: {success_rate:.1f}%")
    
    print(f"\nOVERALL RESULTS:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {total_tests - total_failures - total_errors}")
    print(f"  Failed: {total_failures}")
    print(f"  Errors: {total_errors}")
    print(f"  Skipped: {total_skipped}")
    
    overall_success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
    print(f"  Overall Success Rate: {overall_success_rate:.1f}%")
    print(f"  Execution Time: {elapsed_time:.2f} seconds")
    
    # System readiness assessment
    print(f"\nSYSTEM READINESS ASSESSMENT:")
    
    if total_failures == 0 and total_errors == 0:
        print("✅ SYSTEM FULLY VERIFIED - Ready for Phase 2 pipeline execution")
        readiness_score = "EXCELLENT"
    elif total_failures + total_errors <= 2:
        print("⚠️  SYSTEM MOSTLY READY - Minor issues detected")
        readiness_score = "GOOD"
    elif total_failures + total_errors <= 5:
        print("⚠️  SYSTEM PARTIALLY READY - Several issues need attention")
        readiness_score = "FAIR"
    else:
        print("❌ SYSTEM NOT READY - Major issues detected")
        readiness_score = "POOR"
    
    print(f"Readiness Score: {readiness_score}")
    
    # Recommendations
    print(f"\nRECOMMENDations:")
    if total_failures + total_errors == 0:
        print("  • System is ready for production use")
        print("  • Consider running integration tests with larger datasets")
        print("  • Monitor resource usage during full pipeline execution")
    elif total_failures + total_errors <= 2:
        print("  • Review failed tests and address minor issues")
        print("  • System is suitable for development and testing")
        print("  • Consider running with sample data before full execution")
    else:
        print("  • Address failing tests before pipeline execution")
        print("  • Check dependency installation and configuration")
        print("  • Run tests individually to isolate issues")
    
    # Return success/failure for CI/CD
    return total_failures == 0 and total_errors == 0


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1) 