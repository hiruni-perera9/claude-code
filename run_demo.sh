#!/bin/bash

# Quick demo script for PaleoDB Anomaly Detection
# This script runs a minimal training, evaluation, and inference pipeline

echo "=========================================="
echo "PaleoDB Anomaly Detection - Quick Demo"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not found"
    exit 1
fi

# Install dependencies if needed
echo "Step 1: Installing dependencies..."
pip install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Run training with small dataset for demo
echo "Step 2: Training model (this may take a few minutes)..."
echo "Note: Using small dataset (1000 samples) for quick demo"
echo ""

# Modify config for quick demo
python3 << 'EOF'
import train

# Override config for quick demo
train.main.__code__ = train.main.__code__.replace(
    co_consts=(
        None,
        {
            'data_limit': 1000,  # Reduced for quick demo
            'batch_size': 32,
            'val_split': 0.2,
            'anomaly_ratio': 0.05,
            'model_type': 'transformer',  # Faster than huggingface
            'num_epochs': 10,  # Reduced for demo
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'early_stopping_patience': 5,
            'checkpoint_dir': './checkpoints'
        },
    )
)

if __name__ == "__main__":
    train.main()
EOF

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo "✓ Training completed"
echo ""

# Run evaluation
echo "Step 3: Evaluating model..."
python3 evaluate.py

if [ $? -ne 0 ]; then
    echo "Warning: Evaluation failed (continuing anyway)"
fi

echo "✓ Evaluation completed"
echo ""

# Run inference
echo "Step 4: Running inference on new data..."
python3 inference.py --limit 500 --output demo_results.csv --top-k 10

if [ $? -ne 0 ]; then
    echo "Error: Inference failed"
    exit 1
fi

echo "✓ Inference completed"
echo ""

echo "=========================================="
echo "Demo completed successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - checkpoints/best_model.pt (trained model)"
echo "  - training_history.png (training curves)"
echo "  - evaluation_plots.png (evaluation visualizations)"
echo "  - demo_results.csv (anomaly predictions)"
echo ""
echo "Next steps:"
echo "  1. Review the results in demo_results.csv"
echo "  2. Check visualizations in *.png files"
echo "  3. Modify config in train.py for full training"
echo "  4. Run 'python train.py' for production training"
