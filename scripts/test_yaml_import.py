#!/usr/bin/env python3
"""
Simple test to verify critical dependencies can be imported successfully.
This helps validate that the CI environment is set up correctly.
"""

import sys
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """
    Test if a module can be imported successfully.
    
    Args:
        module_name: Name of the module to import
        package_name: Optional package name for better error messages
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        __import__(module_name)
        logger.info(f"✅ Successfully imported {package_name or module_name}")
        return True, ""
    except ImportError as e:
        error_msg = f"❌ Failed to import {package_name or module_name}: {e}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"❌ Unexpected error importing {package_name or module_name}: {e}"
        logger.error(error_msg)
        return False, error_msg


def main():
    """
    Test importing critical dependencies for MICAP.
    """
    logger.info("🔍 Testing critical dependency imports...")
    
    # Critical dependencies that were failing in CI
    critical_deps = [
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "scikit-learn"),
        ("spacy", "spaCy"),
        ("nltk", "NLTK"),
        ("textblob", "TextBlob"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tqdm", "tqdm"),
        ("pyspark", "PySpark"),
    ]
    
    # Additional ML/DL dependencies
    ml_deps = [
        ("tensorflow", "TensorFlow"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("mlflow", "MLflow"),
    ]
    
    # Development dependencies
    dev_deps = [
        ("pytest", "pytest"),
        ("black", "Black"),
        ("flake8", "Flake8"),
        ("mypy", "MyPy"),
    ]
    
    all_deps = critical_deps + ml_deps + dev_deps
    failed_imports = []
    
    for module_name, package_name in all_deps:
        success, error_msg = test_import(module_name, package_name)
        if not success:
            failed_imports.append((package_name, error_msg))
    
    # Summary
    total_deps = len(all_deps)
    successful_imports = total_deps - len(failed_imports)
    
    logger.info(f"\n📊 Import Test Summary:")
    logger.info(f"   Total packages tested: {total_deps}")
    logger.info(f"   Successful imports: {successful_imports}")
    logger.info(f"   Failed imports: {len(failed_imports)}")
    
    if failed_imports:
        logger.error(f"\n❌ Failed Imports:")
        for package_name, error_msg in failed_imports:
            logger.error(f"   • {package_name}: {error_msg}")
        
        # Exit with error code for CI
        logger.error(f"\n🚨 {len(failed_imports)} critical dependencies failed to import!")
        sys.exit(1)
    else:
        logger.info(f"\n✅ All dependencies imported successfully!")
        
        # Test some basic functionality of critical dependencies
        logger.info(f"\n🧪 Testing basic functionality...")
        
        try:
            import yaml
            test_data = {"test": "data", "number": 42}
            yaml_str = yaml.dump(test_data)
            parsed_data = yaml.safe_load(yaml_str)
            assert parsed_data == test_data
            logger.info("✅ PyYAML basic functionality test passed")
        except Exception as e:
            logger.error(f"❌ PyYAML functionality test failed: {e}")
            sys.exit(1)
        
        try:
            import numpy as np
            arr = np.array([1, 2, 3, 4, 5])
            assert arr.sum() == 15
            logger.info("✅ NumPy basic functionality test passed")
        except Exception as e:
            logger.error(f"❌ NumPy functionality test failed: {e}")
            sys.exit(1)
        
        logger.info(f"\n🎉 All tests passed! Environment is ready for MICAP.")
        sys.exit(0)


if __name__ == "__main__":
    main() 