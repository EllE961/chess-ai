# Autonomous Chess AI System

# Autonomous Chess AI System

An advanced autonomous chess AI agent built from scratch with reinforcement learning, computer vision, and automation capabilities.

## Overview

This system integrates several advanced technologies to create a complete autonomous chess-playing agent:

1. **Deep Reinforcement Learning**: A neural network trained through self-play, inspired by AlphaZero's approach.
2. **Computer Vision**: Detects chess boards and identifies pieces on digital chess platforms.
3. **Automation**: Controls the mouse to execute moves on any online chess website.
4. **MCTS**: Uses Monte Carlo Tree Search guided by the neural network for strong play.

The system can autonomously play chess against humans or other engines on popular websites like Lichess, Chess.com, and Chess24.

## Features

- **Self-improving AI**: The system learns and improves through reinforcement learning without human data.
- **Platform-agnostic**: Works on any digital chess platform with minimal adaptation.
- **Visual Recognition**: Detects chess boards and pieces using sophisticated computer vision.
- **Autonomous Operation**: Fully autonomous play from board detection to move execution.
- **Professional Engine**: Uses residual networks and MCTS similar to professional chess engines.

## System Architecture

The system consists of several modular components:

### Core Components

- **Neural Network**: Deep residual network with policy and value heads
- **MCTS**: Monte Carlo Tree Search for move exploration and evaluation
- **Chess Environment**: Chess rules and board representation

### Vision Components

- **Board Detector**: Detects and extracts chess board from screen
- **Piece Classifier**: Identifies chess pieces using a CNN
- **Position Extractor**: Converts visual data to FEN notation

### Automation Components

- **Move Executor**: Translates chess moves to mouse actions
- **Platform Adapter**: Handles different chess websites' interfaces
- **Calibrator**: Calibrates screen coordinates and board perspective

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Minimum 8GB RAM (16GB+ recommended for training)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/chess-ai.git
cd chess-ai
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create required directories:

```bash
mkdir -p models logs data templates
```

## Usage

### Training the AI

To train the neural network through self-play:

```bash
python train.py --iterations 100
```

Options:

- `--config`: Path to custom config file
- `--resume`: Resume training from the latest checkpoint
- `--iterations`: Number of iterations to train for

### Playing Chess

To play chess on a digital platform:

```bash
python play.py --color white
```

Options:

- `--color`: Choose to play as white, black, or random
- `--manual`: Skip automatic game creation
- `--games`: Number of consecutive games to play

### Calibration

To calibrate the system for a specific chess platform:

```bash
python main.py --mode calibrate
```

## Configuration

The system's behavior can be customized through the configuration files in the `config/` directory:

- **hyperparameters.yaml**: Neural network and training parameters
- **config.py**: System paths and configuration loading

Key parameters:

- **Neural Network**: Layers, filters, and architecture
- **MCTS**: Exploration constant, simulations per move
- **Training**: Batch size, learning rate, self-play games
- **Vision**: Detection thresholds, piece classification confidence
- **Automation**: Mouse movement speed, polling intervals

## Development

### Project Structure

```
chess_ai/
├── config/                      # Configuration files
├── core/                        # Core chess engine
├── training/                    # Training infrastructure
├── vision/                      # Computer vision components
├── automation/                  # Mouse automation
├── utils/                       # Utility functions
├── models/                      # Saved models
├── logs/                        # Log files
├── data/                        # Training data
├── templates/                   # UI templates for detection
├── tests/                       # Unit and integration tests
├── docs/                        # Documentation
├── main.py                      # Main application entry point
├── train.py                     # Training script
└── play.py                      # Play script
```

### Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Performance

The system's playing strength depends on:

1. **Training time**: More self-play iterations improve strength
2. **Hardware**: More powerful GPUs allow deeper searches
3. **MCTS simulations**: More simulations per move increase strength

With sufficient training (100+ iterations) and 800+ MCTS simulations per move, the system can reach a competitive amateur level (1500-1800 ELO).

## Future Improvements

- **Opening book integration**: Incorporate human opening theory
- **Endgame tablebases**: Perfect play in positions with few pieces
- **Distributed training**: Parallel self-play across multiple machines
- **Over-the-board play**: Support for physical chess boards using cameras
- **Multi-game support**: Play multiple games simultaneously

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The AlphaZero paper for the reinforcement learning approach
- The python-chess library for chess rules implementation
- OpenCV for computer vision capabilities
- PyTorch for deep learning framework
