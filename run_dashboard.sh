#!/bin/bash

# Run PaleoDB Anomaly Detection Dashboard
# This script launches the Streamlit dashboard

echo "=========================================="
echo "PaleoDB Anomaly Detection Dashboard"
echo "=========================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit is not installed."
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Check if model checkpoint exists
if [ ! -d "./checkpoints" ]; then
    echo "Warning: No model checkpoints found."
    echo "Please train a model first by running: python train.py"
    echo ""
    echo "Continuing anyway (dashboard will show limited functionality)..."
    echo ""
fi

echo "Starting dashboard on http://localhost:8501"
echo ""
echo "To stop the dashboard, press Ctrl+C"
echo ""

# Run streamlit
streamlit run dashboard.py --server.port 8501 --server.headless true
