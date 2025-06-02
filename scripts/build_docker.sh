#!/bin/bash

# MICAP Docker Build Script
# Builds production-ready Docker images for different environments

set -e  # Exit on any error

# Configuration
IMAGE_NAME="micap"
VERSION="0.1.0"
REGISTRY="your-registry.com"  # Change this to your actual registry
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
MICAP Docker Build Script

Usage: $0 [OPTIONS]

Options:
    -t, --target     Build target (development|testing|production|minimal) [default: production]
    -e, --env        Environment (dev|staging|prod) [default: prod]
    -r, --registry   Docker registry URL [default: ${REGISTRY}]
    -v, --version    Image version [default: ${VERSION}]
    -p, --push       Push to registry after build
    -l, --latest     Tag as latest
    -c, --clean      Clean up intermediate images after build
    -n, --no-cache   Build without cache
    -h, --help       Show this help message

Examples:
    $0                                   # Build production image
    $0 -t development -p                 # Build and push development image
    $0 -t minimal -l -p                  # Build minimal image, tag as latest, and push
    $0 -t production --no-cache -c       # Build production with no cache and cleanup

Build Targets:
    development    Full development environment with dev tools
    testing        Development + testing capabilities
    production     Optimized production image with gunicorn
    minimal        Minimal production image for smaller deployments

EOF
}

# Parse command line arguments
TARGET="production"
ENVIRONMENT="prod"
PUSH=false
LATEST=false
CLEAN=false
NO_CACHE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -l|--latest)
            LATEST=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -n|--no-cache)
            NO_CACHE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate target
if [[ ! "$TARGET" =~ ^(development|testing|production|minimal)$ ]]; then
    log_error "Invalid target: $TARGET"
    show_usage
    exit 1
fi

# Set image tags
BASE_TAG="${REGISTRY}/${IMAGE_NAME}:${VERSION}-${ENVIRONMENT}"
TARGET_TAG="${BASE_TAG}-${TARGET}"
LATEST_TAG="${REGISTRY}/${IMAGE_NAME}:latest"

# Build options
BUILD_ARGS=(
    "--target" "$TARGET"
    "--build-arg" "BUILD_DATE=${BUILD_DATE}"
    "--build-arg" "GIT_COMMIT=${GIT_COMMIT}"
    "--build-arg" "VERSION=${VERSION}"
    "--tag" "$TARGET_TAG"
)

# Add labels
BUILD_ARGS+=(
    "--label" "org.opencontainers.image.created=${BUILD_DATE}"
    "--label" "org.opencontainers.image.version=${VERSION}"
    "--label" "org.opencontainers.image.revision=${GIT_COMMIT}"
    "--label" "org.opencontainers.image.title=MICAP"
    "--label" "org.opencontainers.image.description=Market Intelligence & Competitor Analysis Platform"
    "--label" "org.opencontainers.image.vendor=Ali Vafaei"
)

if [[ "$NO_CACHE" == true ]]; then
    BUILD_ARGS+=("--no-cache")
fi

# Pre-build checks
log_info "Starting MICAP Docker build process..."
log_info "Target: $TARGET"
log_info "Environment: $ENVIRONMENT"
log_info "Version: $VERSION"
log_info "Git Commit: $GIT_COMMIT"
log_info "Image Tag: $TARGET_TAG"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    log_error "Docker is not running or not accessible"
    exit 1
fi

# Check if we're in the project root
if [[ ! -f "Dockerfile" ]]; then
    log_error "Dockerfile not found. Please run this script from the project root."
    exit 1
fi

# Verify required files exist
REQUIRED_FILES=("requirements.txt" "src/api/main.py" "pyproject.toml")
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        log_error "Required file not found: $file"
        exit 1
    fi
done

# Build the image
log_info "Building Docker image with target: $TARGET"
log_info "Running: docker build ${BUILD_ARGS[*]} ."

if docker build "${BUILD_ARGS[@]}" .; then
    log_success "Successfully built image: $TARGET_TAG"
else
    log_error "Failed to build Docker image"
    exit 1
fi

# Tag as latest if requested
if [[ "$LATEST" == true ]]; then
    log_info "Tagging as latest: $LATEST_TAG"
    if docker tag "$TARGET_TAG" "$LATEST_TAG"; then
        log_success "Successfully tagged as latest"
    else
        log_error "Failed to tag as latest"
        exit 1
    fi
fi

# Push to registry if requested
if [[ "$PUSH" == true ]]; then
    log_info "Pushing to registry..."
    
    # Push main tag
    if docker push "$TARGET_TAG"; then
        log_success "Successfully pushed: $TARGET_TAG"
    else
        log_error "Failed to push: $TARGET_TAG"
        exit 1
    fi
    
    # Push latest tag if it was created
    if [[ "$LATEST" == true ]]; then
        if docker push "$LATEST_TAG"; then
            log_success "Successfully pushed: $LATEST_TAG"
        else
            log_error "Failed to push: $LATEST_TAG"
            exit 1
        fi
    fi
fi

# Clean up intermediate images if requested
if [[ "$CLEAN" == true ]]; then
    log_info "Cleaning up intermediate images..."
    
    # Remove dangling images
    if docker image prune -f >/dev/null 2>&1; then
        log_success "Cleaned up dangling images"
    else
        log_warning "Failed to clean up dangling images"
    fi
    
    # Remove build cache
    if docker builder prune -f >/dev/null 2>&1; then
        log_success "Cleaned up build cache"
    else
        log_warning "Failed to clean up build cache"
    fi
fi

# Show image information
log_info "Image build complete!"
echo
docker images --filter "reference=${REGISTRY}/${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Show next steps
echo
log_info "Next steps:"
echo "  • Test the image: docker run --rm -p 8000:8000 $TARGET_TAG"
echo "  • Check health: curl http://localhost:8000/health"
echo "  • View API docs: http://localhost:8000/docs"

if [[ "$PUSH" == false ]]; then
    echo "  • Push to registry: docker push $TARGET_TAG"
fi

log_success "Build process completed successfully!" 