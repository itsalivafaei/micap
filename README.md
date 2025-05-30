# MICAP - Market Intelligence & Competitor Analysis Platform

[![CI/CD Pipeline](https://github.com/your-org/micap/workflows/MICAP%20CI/CD%20Pipeline/badge.svg)](https://github.com/your-org/micap/actions)
[![Documentation](https://github.com/your-org/micap/workflows/Documentation%20CI/badge.svg)](https://github.com/your-org/micap/actions)
[![Code Coverage](https://codecov.io/gh/your-org/micap/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/micap)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning platform for market intelligence and competitor analysis using distributed computing with Apache Spark, advanced NLP, and deep learning models.

## 🚀 Features

- **Distributed Processing**: Scalable data processing with Apache Spark
- **Advanced NLP**: Multi-model sentiment analysis (VADER, TextBlob, BERT, custom models)
- **Deep Learning**: LSTM, CNN, and Transformer models for sentiment classification
- **Real-time Analytics**: Temporal trend analysis and anomaly detection
- **Competitor Analysis**: Brand recognition and competitive intelligence
- **MLOps Ready**: Integrated with MLflow for model tracking and deployment
- **Modern Architecture**: FastAPI backend, containerized deployment, monitoring

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [API Documentation](#-api-documentation)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

## ⚡ Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/micap.git
cd micap

# Start the complete stack
docker-compose up -d

# Access the applications
# - API: http://localhost:8000
# - Spark UI: http://localhost:8080
# - Jupyter: http://localhost:8888
# - Grafana: http://localhost:3000
```

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Download required models
python -m spacy download en_core_web_sm

# Run the pipeline
python scripts/run_phase1_pipeline.py
```

## 🛠 Installation

### Prerequisites

- Python 3.11+
- Java 11+ (for Spark)
- Docker & Docker Compose (optional)
- 8GB+ RAM recommended

### System Dependencies

**macOS:**
```bash
brew install openjdk@11 python@3.11
```

**Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install openjdk-11-jdk python3.11 python3.11-venv
```

### Python Environment

```bash
# Clone repository
git clone https://github.com/your-org/micap.git
cd micap

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

## 📖 Usage

### Basic Pipeline Execution

```python
from src.spark.preprocessing import TextPreprocessor
from src.ml.sentiment_models import ModelEvaluator
from config.spark_config import create_spark_session

# Initialize Spark session
spark = create_spark_session("MICAP-Analysis")

# Load and preprocess data
preprocessor = TextPreprocessor(spark)
df = preprocessor.load_and_preprocess("data/raw/tweets.csv")

# Train and evaluate models
evaluator = ModelEvaluator(spark)
results = evaluator.evaluate_all_models(train_df, test_df, feature_cols)
```

### API Usage

```python
import requests

# Analyze sentiment
response = requests.post("http://localhost:8000/analyze", 
                        json={"text": "Great product! Love it!"})
print(response.json())

# Get trend analysis
response = requests.get("http://localhost:8000/trends/sentiment")
print(response.json())
```

### Running Specific Components

```bash
# Train all models
python scripts/train_all_models.py

# Run competitor analysis
python src/spark/competitor_analysis.py

# Performance benchmarking
python scripts/benchmark_performance.py

# Start API server
uvicorn src.api.main:app --reload
```

## 🏗 Architecture

```
MICAP/
├── src/
│   ├── spark/              # Distributed data processing
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── temporal_analysis.py
│   │   └── competitor_analysis.py
│   ├── ml/                 # Machine learning models
│   │   ├── sentiment_models.py
│   │   ├── deep_learning_models.py
│   │   └── entity_recognition.py
│   ├── api/                # FastAPI backend
│   └── utils/              # Shared utilities
├── scripts/                # Pipeline execution scripts
├── tests/                  # Comprehensive test suite
├── config/                 # Configuration files
└── data/                   # Data storage
```

### Key Components

- **Data Ingestion**: Scalable data loading and validation
- **Preprocessing**: Text cleaning, tokenization, feature engineering
- **Sentiment Analysis**: Multiple model approaches (traditional ML + deep learning)
- **Temporal Analysis**: Time series analysis with anomaly detection
- **Competitor Analysis**: Brand recognition and competitive intelligence
- **API Layer**: RESTful API for real-time analysis
- **Monitoring**: Prometheus metrics and Grafana dashboards

## 📚 API Documentation

The API provides comprehensive endpoints for sentiment analysis and market intelligence:

### Endpoints

- `POST /analyze` - Analyze text sentiment
- `GET /trends/sentiment` - Get sentiment trends
- `GET /competitors/{brand}` - Competitor analysis
- `POST /batch/analyze` - Batch processing
- `GET /health` - Health check

### Interactive Documentation

When running the API, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔧 Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Run linting
flake8 src/ tests/ scripts/
pylint src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suites
pytest tests/unit/test_sentiment_models.py
pytest tests/integration/

# Run performance tests
python scripts/benchmark_performance.py
```

### Code Quality

The project enforces high code quality standards:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **pylint** for advanced static analysis
- **mypy** for type checking
- **bandit** for security scanning

## 🚀 Deployment

### Docker Deployment

```bash
# Build production image
docker build --target production -t micap:latest .

# Run with docker-compose
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=micap
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Environment (dev/staging/prod) | `development` |
| `DATABASE_URL` | PostgreSQL connection string | - |
| `REDIS_URL` | Redis connection string | - |
| `SPARK_MASTER_URL` | Spark master URL | `local[*]` |

## 🧪 Testing

The project includes comprehensive testing at multiple levels:

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component testing
- **Performance Tests**: Benchmarking and load testing
- **End-to-End Tests**: Full pipeline testing

### Test Coverage

Current test coverage: **85%+**

```bash
# Generate coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## 📊 Monitoring

### Metrics Collection

- **Application Metrics**: Request latency, throughput, error rates
- **ML Metrics**: Model accuracy, prediction confidence, drift detection
- **Infrastructure Metrics**: CPU, memory, disk usage
- **Business Metrics**: Sentiment trends, competitor insights

### Dashboards

Access monitoring dashboards:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Spark UI: http://localhost:8080

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Write comprehensive docstrings
- Add tests for new functionality
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Apache Spark community for distributed computing framework
- Hugging Face for transformer models
- spaCy for NLP processing
- TensorFlow and PyTorch teams for deep learning frameworks

## 📞 Support

- **Documentation**: [https://micap.readthedocs.io](https://micap.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-org/micap/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/micap/discussions)
- **Email**: team@micap.io

---

**Built with ❤️ by the MICAP Team**
