#!/bin/bash
# Quick setup script for CSCLog preprocessing on server
# Run this after uploading files to server

echo "=========================================="
echo "CSCLog Preprocessing - Server Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python --version 2>&1)
echo "  $python_version"

if ! command -v python &> /dev/null; then
    echo "  ❌ Python not found!"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  ⚠ venv already exists, skipping..."
else
    python -m venv venv
    echo "  ✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "  ✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "  ✓ pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "  This may take several minutes..."
pip install -r requirements_preprocessing.txt

if [ $? -eq 0 ]; then
    echo "  ✓ Dependencies installed"
else
    echo "  ❌ Failed to install dependencies"
    exit 1
fi

# Check CUDA
echo ""
echo "Checking CUDA availability..."
python -c "import torch; print('  CUDA available:', torch.cuda.is_available()); print('  CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Check GPU
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
        echo "  GPU: $line"
    done
else
    echo "  ⚠ nvidia-smi not found (CPU mode will be used)"
fi

# Create output directory
echo ""
echo "Creating output directory..."
mkdir -p dataset/processed
echo "  ✓ dataset/processed/ created"

# Run validation
echo ""
echo "Running setup validation..."
python test_server_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Setup complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Verify config.yaml settings"
    echo "  2. Ensure dataset/data_full.jsonl exists"
    echo "  3. Run dry-run: python run_preprocessing.py --dry-run"
    echo "  4. Run preprocessing: python run_preprocessing.py"
    echo ""
    echo "Monitor GPU: watch -n 1 nvidia-smi"
else
    echo ""
    echo "=========================================="
    echo "❌ Setup validation failed"
    echo "=========================================="
    echo "Please check errors above and fix before running preprocessing"
    exit 1
fi
