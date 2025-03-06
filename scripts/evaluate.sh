#!/bin/bash
# Evaluation script for the chess AI system

# Exit on error
set -e

# Script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CONFIG="$PROJECT_ROOT/config/hyperparameters.yaml"
MODEL1="$PROJECT_ROOT/models/best_model.pt"
MODEL2=""
GAMES=40
LOG_DIR="$PROJECT_ROOT/logs/evaluation_logs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config)
            CONFIG="$2"
            shift
            shift
            ;;
        --model1)
            MODEL1="$2"
            shift
            shift
            ;;
        --model2)
            MODEL2="$2"
            shift
            shift
            ;;
        --games)
            GAMES="$2"
            shift
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

# Check if model2 is specified
if [ -z "$MODEL2" ]; then
    echo "Error: Model 2 must be specified with --model2"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="$LOG_DIR/evaluation_$TIMESTAMP.log"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

echo "Starting model evaluation at $(date)"
echo "Configuration: $CONFIG"
echo "Model 1: $MODEL1"
echo "Model 2: $MODEL2"
echo "Number of games: $GAMES"
echo "Logging to: $LOG_FILE"

# Run the evaluation
python -c "
import os
import sys
sys.path.append('$PROJECT_ROOT')
from config.config import load_config
from training.evaluate import ModelEvaluator

config = load_config('$CONFIG')
evaluator = ModelEvaluator(config)
evaluator.eval_games = $GAMES

result = evaluator.evaluate('$MODEL1', '$MODEL2')
print(f'Evaluation result: {result:.4f}')
print(f'Model 1 performance: {result * 100:.1f}%')
" 2>&1 | tee "$LOG_FILE"

echo "Evaluation completed at $(date)"