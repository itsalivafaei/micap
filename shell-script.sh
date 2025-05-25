#!/bin/bash

# Navigate to your project directory
cd /Users/ali/Documents/Projects/micap

# Activate your virtual environment
source .venv2/bin/activate

echo "=== DIAGNOSING NUMPY ISSUES ==="

# Check current Python path
echo "Current Python executable:"
which python
echo ""

# Check numpy installation and location
echo "Numpy installation check:"
python -c "import numpy; print(f'Numpy version: {numpy.__version__}'); print(f'Numpy location: {numpy.__file__}')"
echo ""

# Check for conflicting numpy directories
echo "Checking for numpy directories in project:"
find . -name "numpy" -type d 2>/dev/null || echo "No numpy directories found in project"
echo ""

# Check PYTHONPATH
echo "Current PYTHONPATH:"
echo $PYTHONPATH
echo ""

echo "=== FIXING NUMPY ISSUES ==="

# Uninstall and reinstall numpy to fix corruption
echo "Reinstalling numpy..."
pip uninstall numpy -y
pip install numpy==1.26.4

echo ""
echo "Reinstalling pyspark to ensure compatibility..."
pip uninstall pyspark -y
pip install pyspark==4.0.0

echo ""
echo "Verifying installation..."
python -c "import numpy; import pyspark; print('✓ Both numpy and pyspark imported successfully')"

echo ""
echo "=== CLEANUP ==="
# Clear Python cache that might contain corrupted bytecode
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "✓ Cleanup completed"
echo ""
echo "Try running your pipeline again!"