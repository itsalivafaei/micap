#!/usr/bin/env python3
"""
Documentation build script for MICAP.
Handles Sphinx configuration and builds API documentation.
"""

import os
import sys
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sphinx_config():
    """Create Sphinx configuration file."""
    config_content = '''
import os
import sys

# Add the source directory to Python path
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'MICAP'
copyright = '2025, Ali Vafaei'
author = 'Ali Vafaei'
release = '1.0'
version = '1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Mock imports for packages that might not be available during doc build
autodoc_mock_imports = [
    'pyspark',
    'tensorflow', 
    'torch',
    'transformers',
    'spacy',
    'sklearn',
    'mlflow',
    'prophet',
    'vaderSentiment',
]

# HTML theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

# HTML theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Master document
master_doc = 'index'

# Source file suffixes
source_suffix = {
    '.rst': None,
}

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}
'''
    return config_content.strip()


def create_index_rst():
    """Create main index.rst file."""
    index_content = '''
Welcome to MICAP Documentation
==============================

Market Intelligence & Competitor Analysis Platform (MICAP) is a comprehensive solution for sentiment analysis and competitive intelligence using advanced machine learning and distributed computing.

Features
--------

* **Scalable Data Processing**: Built on Apache Spark for handling large-scale data
* **Multi-Model Sentiment Analysis**: Traditional ML and deep learning approaches  
* **Real-time API**: FastAPI-based REST API for real-time analysis
* **Temporal Analysis**: Time series analysis with anomaly detection
* **Competitor Intelligence**: Brand recognition and competitive analysis
* **Comprehensive Monitoring**: Prometheus metrics and Grafana dashboards

Quick Start
-----------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/micap.git
   cd micap

   # Install dependencies
   pip install -r requirements.txt

   # Run the training pipeline
   python scripts/train_all_models.py

   # Start the API server
   uvicorn src.api.main:app --reload

API Documentation
-----------------

The MICAP API provides comprehensive endpoints for sentiment analysis and market intelligence:

* ``POST /analyze`` - Analyze text sentiment
* ``GET /trends/sentiment`` - Get sentiment trends  
* ``GET /competitors/{brand}`` - Competitor analysis
* ``POST /batch/analyze`` - Batch processing
* ``GET /health`` - Health check

Architecture Overview
--------------------

MICAP is built with a modular architecture:

* **src/spark/**: Distributed data processing components
* **src/ml/**: Machine learning models and evaluation
* **src/api/**: FastAPI backend services
* **src/utils/**: Shared utilities and helpers
* **scripts/**: Pipeline execution and automation
* **tests/**: Comprehensive test suite

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
'''
    return index_content.strip()


def create_custom_css():
    """Create custom CSS for documentation styling."""
    css_content = '''
/* Custom styling for MICAP documentation */

.wy-nav-content {
    max-width: 1200px;
}

.rst-content .section > img {
    margin-bottom: 24px;
}

.rst-content .admonition-title {
    font-weight: bold;
}

/* Code block styling */
.rst-content .highlight {
    margin: 1px 0 24px 0;
}

/* Table styling */
.rst-content table.docutils {
    border: 1px solid #e1e4e5;
}

.rst-content table.docutils td {
    border: 1px solid #e1e4e5;
    padding: 12px;
}

/* API documentation styling */
.rst-content dl:not(.docutils) dt {
    background: #f8f9fa;
    border-left: 3px solid #2980b9;
    padding: 6px;
    margin-bottom: 6px;
}
'''
    return css_content.strip()


def setup_docs_directory():
    """Set up the docs directory with proper structure."""
    docs_dir = Path("docs")
    
    # Create docs directory if it doesn't exist
    docs_dir.mkdir(exist_ok=True)
    
    # Create _static directory
    static_dir = docs_dir / "_static"
    static_dir.mkdir(exist_ok=True)
    
    # Write configuration files
    logger.info("Creating Sphinx configuration...")
    
    with open(docs_dir / "conf.py", "w") as f:
        f.write(create_sphinx_config())
    
    with open(docs_dir / "index.rst", "w") as f:
        f.write(create_index_rst())
        
    with open(static_dir / "custom.css", "w") as f:
        f.write(create_custom_css())
    
    logger.info("Documentation structure created successfully")


def generate_api_docs():
    """Generate API documentation using sphinx-apidoc."""
    logger.info("Generating API documentation...")
    
    # Change to docs directory
    os.chdir("docs")
    
    # Run sphinx-apidoc
    import subprocess
    result = subprocess.run([
        "sphinx-apidoc", "-o", ".", "../src", 
        "--force", "--separate", "--module-first"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"sphinx-apidoc failed: {result.stderr}")
        return False
        
    logger.info("API documentation generated successfully")
    return True


def build_html_docs():
    """Build HTML documentation."""
    logger.info("Building HTML documentation...")
    
    import subprocess
    result = subprocess.run([
        "sphinx-build", "-b", "html", ".", "_build/html"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Sphinx build failed: {result.stderr}")
        logger.error(f"Stdout: {result.stdout}")
        return False
        
    logger.info("HTML documentation built successfully")
    return True


def check_dependencies():
    """Check if required documentation tools are available."""
    import shutil
    
    required_tools = ['sphinx-apidoc', 'sphinx-build']
    missing_tools = []
    
    for tool in required_tools:
        if not shutil.which(tool):
            missing_tools.append(tool)
    
    if missing_tools:
        logger.error(f"Missing required tools: {', '.join(missing_tools)}")
        logger.error("Please install Sphinx: pip install sphinx sphinx-rtd-theme")
        return False
    
    return True


def main():
    """Main function to build documentation."""
    logger.info("Starting MICAP documentation build...")
    
    try:
        # Check dependencies first
        if not check_dependencies():
            logger.error("Missing dependencies. Please install required packages.")
            sys.exit(1)
        
        # Setup documentation directory
        setup_docs_directory()
        
        # Generate API documentation
        if not generate_api_docs():
            sys.exit(1)
            
        # Build HTML documentation
        if not build_html_docs():
            sys.exit(1)
            
        logger.info("Documentation build completed successfully!")
        logger.info("Documentation available at: docs/_build/html/index.html")
        
    except Exception as e:
        logger.error(f"Documentation build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 