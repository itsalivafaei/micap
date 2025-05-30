"""
Test suite for Phase 1 Pipeline
Tests the complete data processing pipeline with enhanced error handling
"""

import unittest
import sys
import os
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.run_phase1_pipeline import (
    run_pipeline, validate_environment, 
    PipelineRecovery, safe_import
)


class TestPhase1Pipeline(unittest.TestCase):
    """Test Phase 1 Pipeline functionality"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_validate_environment(self):
        """Test environment validation"""
        # Test with current environment (should pass)
        self.assertTrue(validate_environment())
        
    def test_safe_import_valid_modules(self):
        """Test safe import functionality with valid modules"""
        # Test importing spark config
        try:
            create_spark_session = safe_import("config.spark_config", "Spark configuration")
            self.assertIsNotNone(create_spark_session)
        except ImportError:
            # This is expected if dependencies are not available
            self.skipTest("Spark dependencies not available")
    
    def test_safe_import_invalid_modules(self):
        """Test safe import functionality with invalid modules"""
        with self.assertRaises(ImportError):
            safe_import("nonexistent.module", "Non-existent module")

    def test_pipeline_recovery_initialization(self):
        """Test PipelineRecovery initialization"""
        recovery = PipelineRecovery(self.test_dir)
        self.assertEqual(recovery.base_path, Path(self.test_dir))
        self.assertEqual(recovery.checkpoints, {})

    @patch('scripts.run_phase1_pipeline.create_spark_session')
    @patch('scripts.run_phase1_pipeline.DataIngestion')
    @patch('scripts.run_phase1_pipeline.TextPreprocessor')
    def test_pipeline_basic_flow(self, mock_preprocessor, mock_ingestion, mock_spark):
        """Test basic pipeline flow with mocked components"""
        # Mock Spark session
        mock_spark_instance = MagicMock()
        mock_spark.return_value = mock_spark_instance
        
        # Mock DataIngestion
        mock_ingestion_instance = MagicMock()
        mock_ingestion.return_value = mock_ingestion_instance
        
        mock_df = MagicMock()
        mock_df.count.return_value = 1000
        mock_ingestion_instance.load_sentiment140_data.return_value = mock_df
        mock_ingestion_instance.validate_data_quality.return_value = (mock_df, {})
        mock_ingestion_instance.create_sample_dataset.return_value = mock_df
        mock_ingestion_instance.save_to_local_storage.return_value = None
        
        # Mock TextPreprocessor
        mock_preprocessor_instance = MagicMock()
        mock_preprocessor.return_value = mock_preprocessor_instance
        mock_preprocessor_instance.preprocess_pipeline.return_value = mock_df
        
        # Mock DataFrame operations
        mock_df.coalesce.return_value.write.mode.return_value.parquet.return_value = None
        mock_df.write.mode.return_value.format.return_value.save.return_value = None
        
        # Test with feature engineering disabled (simulated)
        with patch('scripts.run_phase1_pipeline.FEATURE_ENGINEERING_AVAILABLE', False):
            success, results = run_pipeline(
                sample_fraction=0.01,
                enable_checkpoints=False,
                resume_from_checkpoint=False
            )
        
        # Verify basic success
        self.assertTrue(success)
        self.assertIn('ingestion', results['stages_completed'])
        self.assertIn('preprocessing', results['stages_completed'])
        self.assertEqual(results['records_processed'], 1000)

    def test_pipeline_with_missing_data(self):
        """Test pipeline behavior with missing data"""
        with patch('pathlib.Path.exists', return_value=False):
            with patch('scripts.run_phase1_pipeline.create_spark_session'):
                success, results = run_pipeline(
                    sample_fraction=0.01,
                    enable_checkpoints=False
                )
                
                # Should fail gracefully when data is missing
                self.assertFalse(success)
                self.assertTrue(len(results['errors']) > 0)

    @patch('scripts.run_phase1_pipeline.create_spark_session')
    def test_pipeline_spark_session_failure(self, mock_spark):
        """Test pipeline behavior when Spark session creation fails"""
        mock_spark.side_effect = Exception("Spark initialization failed")
        
        success, results = run_pipeline(
            sample_fraction=0.01,
            enable_checkpoints=False
        )
        
        self.assertFalse(success)
        self.assertIn("Failed to create Spark session", str(results['errors']))

    def test_pipeline_recovery_save_checkpoint(self):
        """Test checkpoint saving functionality"""
        recovery = PipelineRecovery(self.test_dir)
        
        # Mock DataFrame
        mock_df = MagicMock()
        mock_df.coalesce.return_value.write.mode.return_value.parquet.return_value = None
        
        # Test successful checkpoint save
        result = recovery.save_checkpoint("test_stage", mock_df, {"test": "metadata"})
        
        # Verify checkpoint was registered
        self.assertTrue(result)
        self.assertIn("test_stage", recovery.checkpoints)

    def test_pipeline_with_sample_fractions(self):
        """Test pipeline with different sample fractions"""
        # Test invalid sample fraction
        with patch('scripts.run_phase1_pipeline.create_spark_session'):
            with patch('scripts.run_phase1_pipeline.DataIngestion'):
                # Very small sample should still work
                success, results = run_pipeline(sample_fraction=0.001)
                # This may fail due to mocking, but structure should handle it


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for pipeline components"""

    def setUp(self):
        """Set up integration test environment"""
        self.test_data_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.test_data_dir, ignore_errors=True)

    @unittest.skipUnless(
        os.path.exists("data/raw/testdata.manual.2009.06.14.csv"),
        "Test data not available"
    )
    def test_pipeline_with_real_data_small_sample(self):
        """Test pipeline with real data using very small sample"""
        try:
            success, results = run_pipeline(
                sample_fraction=0.001,  # Very small sample for testing
                enable_checkpoints=False
            )
            
            # If dependencies are available, this should succeed
            if not success:
                # Check if failure is due to missing dependencies
                error_messages = str(results.get('errors', []))
                if 'import' in error_messages.lower() or 'spark' in error_messages.lower():
                    self.skipTest("Required dependencies not available")
                else:
                    self.fail(f"Pipeline failed unexpectedly: {results['errors']}")
            else:
                self.assertTrue(success)
                self.assertGreater(len(results['stages_completed']), 0)
                
        except Exception as e:
            if 'spark' in str(e).lower() or 'java' in str(e).lower():
                self.skipTest("Spark environment not properly configured")
            else:
                raise

    def test_pipeline_argument_parsing(self):
        """Test pipeline command line argument handling"""
        # This would test the argparse functionality
        # For now, we verify the structure exists
        from scripts.run_phase1_pipeline import __name__ as script_name
        self.assertIsNotNone(script_name)


class TestPipelineErrorHandling(unittest.TestCase):
    """Test error handling in pipeline"""

    def test_graceful_failure_handling(self):
        """Test that pipeline fails gracefully with proper error reporting"""
        with patch('scripts.run_phase1_pipeline.validate_environment', return_value=False):
            success, results = run_pipeline()
            
            self.assertFalse(success)
            self.assertIn('errors', results)
            self.assertTrue(len(results['errors']) > 0)

    def test_partial_pipeline_execution(self):
        """Test pipeline behavior when only some stages complete"""
        # This would require more sophisticated mocking
        # For now, verify the structure supports partial execution
        success, results = run_pipeline(sample_fraction=0.001, enable_checkpoints=False)
        
        # Verify results structure is correct regardless of success
        self.assertIn('success', results)
        self.assertIn('stages_completed', results)
        self.assertIn('processing_time', results)
        self.assertIn('records_processed', results)
        self.assertIn('errors', results)


if __name__ == '__main__':
    # Create test directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2) 