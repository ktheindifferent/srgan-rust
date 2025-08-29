#!/bin/bash

echo "Starting quick training session..."
echo "Training will run for 200 steps and then save the model."
echo ""

# Run training in background
./target/release/srgan-rust train ./training_data/ ./my_model.rsr \
    --batch_size 1 \
    --log_depth 2 \
    --width 8 \
    -q &

# Get the PID
TRAIN_PID=$!

# Monitor for 200 steps (approximately 30 seconds)
echo "Training in progress..."
sleep 30

# Send interrupt signal to save the model
echo ""
echo "Sending interrupt to save model..."
kill -INT $TRAIN_PID

# Wait for process to finish saving
sleep 2

# Check if model was created
if [ -f "./my_model.rsr" ]; then
    echo "✅ Model saved successfully!"
    ls -lh ./my_model.rsr
else
    echo "❌ Model not saved. Training may need more time."
fi