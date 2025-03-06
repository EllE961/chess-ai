#!/bin/bash
# Setup script for the chess AI system

# Exit on error
set -e

# Script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up Chess AI system..."
echo "Project root: $PROJECT_ROOT"

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/venv"
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "$PROJECT_ROOT/requirements.txt"

# Create necessary directories
echo "Creating directories..."
mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/data/game_records"
mkdir -p "$PROJECT_ROOT/logs/training_logs"
mkdir -p "$PROJECT_ROOT/logs/game_logs"
mkdir -p "$PROJECT_ROOT/logs/vision_debug"
mkdir -p "$PROJECT_ROOT/templates"

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e "$PROJECT_ROOT"

echo "Setup complete!"
echo "To activate the environment, run: source $PROJECT_ROOT/venv/bin/activate"