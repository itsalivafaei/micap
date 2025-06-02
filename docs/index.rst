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