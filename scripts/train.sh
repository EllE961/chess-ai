#!/bin/bash
# Training script for the chess AI system

# Exit on error
set -e

# Script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CONFIG="$PROJECT_ROOT/config/hyperparameters.yaml"
ITERATIONS=0
RESUME=false
LOG_DIR="$PROJECT_ROOT/logs/training_logs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config)
            CONFIG="$2"
            shift
            shift
            ;;
        --iterations)
            ITERATIONS="$2"
            shift
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

echo "Starting training at $(date)"
echo "Configuration: $CONFIG"
echo "Logging to: $LOG_FILE"

# Build the command
CMD="python $PROJECT_ROOT/train.py --config $CONFIG"

if [ "$RESUME" = true ]; then
    CMD="$CMD --resume"
fi

if [ "$ITERATIONS" -gt 0 ]; then
    CMD="$CMD --iterations $ITERATIONS"
fi

# Run the training
echo "Running command: $CMD"
echo "Training logs will be saved to: $LOG_FILE"
$CMD 2>&1 | tee "$LOG_FILE"

echo "Training completed at $(date)"