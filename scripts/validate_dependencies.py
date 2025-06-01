#!/usr/bin/env python3
"""
MICAP Dependency Validation Script

This script validates that all project dependencies can be installed correctly
and imports work as expected. Use this before pushing changes to catch
dependency issues early.

Usage:
    python scripts/validate_dependencies.py
"""

import subprocess
import sys
import importlib
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Critical dependencies to validate
CRITICAL_IMPORTS = [
    'numpy',
    'pandas',
    'sklearn',
    'nltk',
    'spacy',
    'tensorflow',
    'torch',
    'transformers',
    'matplotlib',
    'seaborn',
    'pyspark',
    'mlflow',
    'yaml',  # PyYAML
    'fuzzywuzzy',
    'prophet',
    'statsmodels'
]

def run_command(command: str) -> Tuple[bool, str, str]:
    """
    Execute a shell command and return success status and outputs.
    
    Args:
        command: Shell command to execute
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def validate_installation() -> bool:
    """
    Validate that pip install works correctly with our configuration.
    
    Returns:
        True if installation succeeds, False otherwise
    """
    logger.info("Validating dependency installation...")
    
    # Test installation sequence that matches CI
    commands = [
        "python -m pip install --upgrade pip setuptools wheel",
        "pip install cython",
        "pip check"  # Validate no dependency conflicts
    ]
    
    for command in commands:
        logger.info(f"Running: {command}")
        success, stdout, stderr = run_command(command)
        
        if not success:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error: {stderr}")
            return False
        else:
            logger.info("✓ Command succeeded")
    
    return True

def validate_imports() -> bool:
    """
    Validate that critical packages can be imported successfully.
    
    Returns:
        True if all imports succeed, False otherwise
    """
    logger.info("Validating critical package imports...")
    
    failed_imports = []
    
    for package in CRITICAL_IMPORTS:
        try:
            importlib.import_module(package)
            logger.info(f"✓ Successfully imported {package}")
        except ImportError as e:
            logger.error(f"✗ Failed to import {package}: {e}")
            failed_imports.append(package)
        except Exception as e:
            logger.error(f"✗ Unexpected error importing {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        logger.error(f"Failed to import {len(failed_imports)} packages: {failed_imports}")
        return False
    
    logger.info(f"✓ All {len(CRITICAL_IMPORTS)} critical imports successful")
    return True

def validate_yaml_specifically() -> bool:
    """
    Specifically test PyYAML functionality since it was causing CI issues.
    
    Returns:
        True if PyYAML works correctly, False otherwise
    """
    logger.info("Validating PyYAML functionality...")
    
    try:
        import yaml
        
        # Test basic YAML operations
        test_data = {
            'test': 'data',
            'numbers': [1, 2, 3],
            'nested': {'key': 'value'}
        }
        
        # Test serialization
        yaml_string = yaml.dump(test_data)
        logger.info("✓ YAML serialization successful")
        
        # Test deserialization  
        parsed_data = yaml.safe_load(yaml_string)
        logger.info("✓ YAML deserialization successful")
        
        # Validate data integrity
        if parsed_data == test_data:
            logger.info("✓ YAML data integrity validated")
            return True
        else:
            logger.error("✗ YAML data integrity check failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ PyYAML validation failed: {e}")
        return False

def main():
    """Main validation routine."""
    logger.info("Starting MICAP dependency validation...")
    
    all_checks_passed = True
    
    # Run validation checks
    checks = [
        ("Installation", validate_installation),
        ("Imports", validate_imports), 
        ("PyYAML", validate_yaml_specifically)
    ]
    
    for check_name, check_func in checks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {check_name} validation...")
        logger.info(f"{'='*50}")
        
        if not check_func():
            logger.error(f"✗ {check_name} validation failed")
            all_checks_passed = False
        else:
            logger.info(f"✓ {check_name} validation passed")
    
    # Final summary
    logger.info(f"\n{'='*50}")
    if all_checks_passed:
        logger.info("🎉 All dependency validations passed!")
        logger.info("Dependencies should work correctly in CI pipeline")
        sys.exit(0)
    else:
        logger.error("❌ Some dependency validations failed")
        logger.error("Please fix the issues before pushing to repository")
        sys.exit(1)

if __name__ == "__main__":
    main() 