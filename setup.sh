#!/bin/bash
# Setup script for PaleoDB Anomaly Detection Dashboard
# Checks for trained model and trains if needed

echo "=========================================="
echo "PaleoDB Dashboard Setup"
echo "=========================================="
echo ""

# Check if checkpoints directory exists
if [ -d "./checkpoints" ] && [ -f "./checkpoints/best_model.pt" ]; then
    echo "✅ Model checkpoint found!"
    echo ""
    ls -lh ./checkpoints/best_model.pt
    echo ""
else
    echo "⚠️  No trained model found."
    echo ""
    echo "Would you like to train a model now? (y/n)"
    echo "Note: Training will take approximately 10-15 minutes"
    echo ""

    # Check if running in non-interactive mode (deployment)
    if [ -t 0 ]; then
        # Interactive mode - ask user
        read -p "Train model? (y/n): " response
    else
        # Non-interactive mode (deployment) - auto-train
        response="y"
        echo "Running in deployment mode - automatically training model..."
    fi

    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo ""
        echo "Installing dependencies..."
        pip install -q -r requirements.txt

        echo ""
        echo "Training model (this will take a few minutes)..."
        echo "Using a smaller dataset for faster training..."

        # Train with smaller dataset for faster setup
        python3 train.py

        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Model trained successfully!"

            # Run evaluation
            echo ""
            echo "Evaluating model..."
            python3 evaluate.py

            if [ $? -eq 0 ]; then
                echo ""
                echo "✅ Evaluation completed!"
            fi
        else
            echo ""
            echo "❌ Training failed. Please check the error messages above."
            exit 1
        fi
    else
        echo ""
        echo "Skipping training. You can train later by running:"
        echo "  python3 train.py"
        echo ""
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start the dashboard, run:"
echo "  ./run_dashboard.sh"
echo ""
