#!/bin/bash
# Deployment setup script for gold price predictor
# This script handles preprocessing and training during deployment

set -e  # Exit on error (but we'll handle gracefully)

echo "=========================================="
echo "Gold Price Predictor - Deployment Setup"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Project root: $PROJECT_ROOT"
echo "Backend directory: $SCRIPT_DIR"

# Change to project root (where backend/ is a subdirectory)
cd "$PROJECT_ROOT"

# Step 1: Preprocessing
echo ""
echo "Step 1: Running preprocessing..."
if python -m backend.services.preprocess; then
    echo "[OK] Preprocessing completed successfully"
else
    echo "[!] Preprocessing failed - this is OK if raw data is not available"
    echo "    The API will start but predictions may not work until data is added"
fi

# Step 2: Training
echo ""
echo "Step 2: Training model..."
if python -m backend.services.trainer --no-tune; then
    echo "[OK] Model training completed successfully"
else
    echo "[!] Training failed - this is OK if processed data is not available"
    echo "    The API will start but predictions will return 503 error"
fi

echo ""
echo "=========================================="
echo "Deployment setup complete!"
echo "=========================================="
echo ""
echo "Next: The API server will start with uvicorn"
echo ""

