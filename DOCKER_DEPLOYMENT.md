# MICAP Docker Deployment Guide

## 🚀 Quick Start for Developers

This guide helps developers quickly get MICAP running using Docker for development and testing purposes.

### Prerequisites

- Docker installed and running
- At least 4GB of available disk space
- 8GB RAM recommended

### 🏗️ Building the Image

We provide multiple build options:

#### Option 1: Simple Development Build (Recommended)
```bash
# Build the lightweight development image
docker build -f Dockerfile.dev -t micap:0.1.0-dev .
```

#### Option 2: Multi-stage Build
```bash
# Build using the development target
docker build --target development -t micap:0.1.0-dev .
```

#### Option 3: Using Build Script
```bash
# Use the provided build script
./scripts/build_docker.sh -t development -v 0.1.0 -r local
```

### 🚢 Running the Container

#### Basic Run
```bash
# Run the API server
docker run --rm -p 8000:8000 micap:0.1.0-dev
```

#### With Environment Variables
```bash
# Run with custom configuration
docker run --rm -p 8000:8000 \
  -e ENV=development \
  -e LOG_LEVEL=debug \
  micap:0.1.0-dev
```

#### With Data Persistence
```bash
# Run with volume mounting for data persistence
docker run --rm -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  micap:0.1.0-dev
```

### 🧪 Testing the API

Once the container is running, you can test the API:

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs

# Test sentiment analysis
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a great product!"}'

# Batch analysis
curl -X POST "http://localhost:8000/batch/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Great product!", "Terrible service", "Okay experience"],
    "include_details": true
  }'

# Sentiment trends
curl "http://localhost:8000/trends/sentiment?days=7"
```

### 🐳 Using Docker Compose

For a complete development setup with database and Redis:

```bash
# Copy environment file
cp env.example .env

# Edit .env with your settings
nano .env

# Start the complete stack
docker-compose up -d

# Or for production-like setup
docker-compose -f docker-compose.prod.yml up -d
```

### 📊 Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and system status |
| `/docs` | GET | Interactive API documentation |
| `/analyze` | POST | Single text sentiment analysis |
| `/batch/analyze` | POST | Batch sentiment analysis |
| `/trends/sentiment` | GET | Historical sentiment trends |

### 🔧 Configuration Options

The container accepts these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `development` | Runtime environment |
| `LOG_LEVEL` | `info` | Logging level |
| `API_PORT` | `8000` | API server port |
| `DATABASE_URL` | - | PostgreSQL connection string |
| `REDIS_URL` | - | Redis connection string |

### 📝 Development Tips

1. **Volume Mounting**: Mount your source code for live development:
   ```bash
   docker run --rm -p 8000:8000 \
     -v $(pwd)/src:/app/src \
     micap:0.1.0-dev
   ```

2. **Debugging**: Run with interactive shell:
   ```bash
   docker run --rm -it -p 8000:8000 micap:0.1.0-dev bash
   ```

3. **Logs**: View container logs:
   ```bash
   docker logs -f <container_id>
   ```

### 🚀 Sharing the Image

#### Export/Import
```bash
# Save image to tar file
docker save micap:0.1.0-dev > micap-0.1.0-dev.tar

# Load image on another machine
docker load < micap-0.1.0-dev.tar
```

#### Registry Push/Pull
```bash
# Tag for registry
docker tag micap:0.1.0-dev your-registry.com/micap:0.1.0-dev

# Push to registry
docker push your-registry.com/micap:0.1.0-dev

# Pull on another machine
docker pull your-registry.com/micap:0.1.0-dev
```

### 🔍 Troubleshooting

#### Common Issues

1. **Port Already in Use**
   ```bash
   # Use different port
   docker run --rm -p 8001:8000 micap:0.1.0-dev
   ```

2. **Memory Issues**
   ```bash
   # Limit memory usage
   docker run --rm -p 8000:8000 --memory="2g" micap:0.1.0-dev
   ```

3. **Permission Issues**
   ```bash
   # Run as current user
   docker run --rm -p 8000:8000 -u $(id -u):$(id -g) micap:0.1.0-dev
   ```

#### Health Check Failed
- Ensure the API is starting properly: `docker logs <container_id>`
- Check if port 8000 is accessible inside the container
- Verify all dependencies are installed correctly

#### Model Loading Issues
- The spaCy model should download automatically during build
- If missing, rebuild the image or manually install in running container

### 🏗️ Build Targets

The multi-stage Dockerfile provides several targets:

- **`development`**: Full development environment with dev tools
- **`testing`**: Includes test dependencies and runs tests
- **`production`**: Optimized for production deployment
- **`minimal`**: Smallest possible image size

### 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [MICAP Project Documentation](./README.md)

### 🤝 Contributing

When sharing images with other developers:

1. Always include this deployment guide
2. Specify the exact version and build target used
3. Include any custom environment variables needed
4. Provide sample API calls for testing

---

**Version**: 0.1.0  
**Last Updated**: 2024-06-02  
**Maintainer**: Ali Vafaei 