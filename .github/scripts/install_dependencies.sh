#!/bin/bash
# Standardized dependency installation script for MICAP CI/CD
# This script ensures proper build order and dependency resolution

set -euo pipefail  # Exit on any error

echo "🔧 Starting MICAP dependency installation..."

# Color output for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to upgrade core tools
upgrade_core_tools() {
    log_info "Upgrading core Python build tools..."
    python -m pip install --upgrade --no-cache-dir pip
    
    # Install build dependencies first (critical for PyYAML and other C extensions)
    log_info "Installing critical build dependencies..."
    pip install --upgrade --no-cache-dir \
        "setuptools>=75.3.0" \
        "wheel>=0.45.1" \
        "cython>=3.0.11"
    
    log_info "Core tools upgraded successfully ✅"
}

# Function to install main dependencies
install_main_deps() {
    log_info "Installing main dependencies from requirements.txt..."
    
    # Install with specific flags to avoid build issues
    pip install --no-cache-dir \
        --prefer-binary \
        --only-binary=:all: \
        -r requirements.txt || {
        log_warn "Binary installation failed, trying with source builds..."
        pip install --no-cache-dir -r requirements.txt
    }
    
    log_info "Main dependencies installed successfully ✅"
}

# Function to install dev dependencies
install_dev_deps() {
    if [[ "${INSTALL_DEV:-true}" == "true" ]]; then
        log_info "Installing development dependencies..."
        
        pip install --no-cache-dir \
            --prefer-binary \
            -r requirements-dev.txt
        
        log_info "Development dependencies installed successfully ✅"
    else
        log_info "Skipping development dependencies (INSTALL_DEV=false)"
    fi
}

# Function to install additional tools
install_additional_tools() {
    local tools="$1"
    if [[ -n "$tools" ]]; then
        log_info "Installing additional tools: $tools"
        pip install --no-cache-dir $tools
        log_info "Additional tools installed successfully ✅"
    fi
}

# Function to download spaCy models
download_spacy_models() {
    if [[ "${DOWNLOAD_SPACY:-true}" == "true" ]]; then
        log_info "Downloading spaCy models..."
        python -m spacy download en_core_web_sm --quiet || {
            log_warn "Failed to download spaCy model, but continuing..."
        }
        log_info "spaCy models downloaded successfully ✅"
    fi
}

# Function to verify installation
verify_installation() {
    log_info "Verifying critical package imports..."
    
    python -c "
import sys
critical_packages = [
    'yaml', 'pandas', 'numpy', 'sklearn', 
    'torch', 'tensorflow', 'nltk', 'spacy'
]

failed_imports = []
for package in critical_packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError as e:
        failed_imports.append(f'{package}: {e}')
        print(f'❌ {package}: {e}')

if failed_imports:
    print(f'\\n⚠️  Failed imports: {len(failed_imports)}')
    sys.exit(1)
else:
    print(f'\\n🎉 All critical packages imported successfully!')
"
}

# Main execution
main() {
    log_info "MICAP Dependency Installation Script v1.0"
    log_info "Python version: $(python --version)"
    log_info "Pip version: $(pip --version)"
    
    # Parse command line arguments
    ADDITIONAL_TOOLS="${1:-}"
    
    # Execute installation steps
    upgrade_core_tools
    install_main_deps
    install_dev_deps
    install_additional_tools "$ADDITIONAL_TOOLS"
    download_spacy_models
    verify_installation
    
    log_info "🎉 MICAP dependency installation completed successfully!"
    
    # Display summary
    echo ""
    echo "📊 Installation Summary:"
    echo "========================"
    pip list | grep -E "(pyyaml|cython|setuptools|wheel|numpy|pandas)" || true
}

# Execute main function with all arguments
main "$@" 