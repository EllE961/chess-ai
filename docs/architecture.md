# Chess AI System Architecture

This document provides a detailed overview of the system architecture, explaining how the different components interact to create a complete autonomous chess AI system.

## System Overview

The Chess AI system is designed with a modular architecture that separates concerns and allows components to be developed, tested, and improved independently. The main components are:

1. **Core Chess Engine**: Handles chess rules, position evaluation, and move selection
2. **Training Infrastructure**: Manages self-play and neural network training
3. **Computer Vision System**: Detects chess boards and pieces on screen
4. **Automation System**: Controls the mouse to interact with chess platforms
5. **Utility Functions**: Provides common functionality used throughout the system

## Component Interaction

Here's how these components interact during typical operations:

### Game Play Workflow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│ Board Detector │────>│Position Extractor│───>│  Chess Engine  │
└────────────────┘     └────────────────┘     └────────────────┘
         │                                              │
         │                                              │
         │                                              ▼
┌────────▼───────┐                            ┌────────────────┐
│Piece Classifier│                            │     MCTS       │
└────────────────┘                            └────────────────┘
                                                       │
                                                       │
                                                       ▼
                                              ┌────────────────┐
                                              │ Neural Network │
                                              └────────────────┘
                                                       │
                                                       │
┌────────────────┐                                     │
│Platform Adapter│<────────────────────────────────────┘
└────────────────┘
         │
         │
         ▼
┌────────────────┐
│ Move Executor  │
└────────────────┘
```

1. **Board Detection**: The `BoardDetector` captures the screen and identifies the chess board
2. **Piece Classification**: The `PieceClassifier` identifies pieces on each square
3. **Position Extraction**: The `PositionExtractor` converts the classified board to a FEN string
4. **Move Selection**: The `MCTS` algorithm uses the `ChessNetwork` to evaluate positions and select the best move
5. **Move Execution**: The `MoveExecutor` translates the move to mouse actions and executes it

### Training Workflow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   Self-Play    │────>│ Replay Buffer  │────>│Neural Network  │
└────────────────┘     └────────────────┘     │    Training    │
       │ ▲                                    └────────────────┘
       │ │                                             │
       │ │                                             │
       │ └─────────────────────────────────────────────┘
       │
       ▼
┌────────────────┐     ┌────────────────┐
│     MCTS       │────>│ Neural Network │
└────────────────┘     └────────────────┘
```

1. **Self-Play**: The system plays games against itself using MCTS and the neural network
2. **Data Collection**: Game states, policies, and outcomes are stored in the replay buffer
3. **Network Training**: The neural network is trained on the collected data
4. **Evaluation**: The new network is evaluated against the previous best
5. **Iteration**: The process repeats with the improved neural network

## Detailed Component Design

### Core Chess Engine

The core chess engine is built around these main classes:

- **ChessEnvironment**: Wraps the python-chess Board class and provides additional functionality for state encoding
- **ChessNetwork**: Implements a deep residual neural network with policy and value heads
- **MCTS**: Implements the Monte Carlo Tree Search algorithm guided by the neural network
- **SelfPlay**: Manages the generation of self-play games for training

### Vision System

The vision system consists of:

- **BoardDetector**: Detects chess boards on screen using contour detection and perspective transformation
- **PieceClassifier**: Uses a convolutional neural network to classify pieces on each square
- **PositionExtractor**: Converts classified squares to FEN notation

### Automation System

The automation system includes:

- **MoveExecutor**: Controls mouse movements to execute chess moves
- **PlatformAdapter**: Handles platform-specific UI elements and interactions
- **Calibrator**: Maps screen coordinates to chess squares

### Training Infrastructure

The training infrastructure consists of:

- **Trainer**: Manages the overall training process
- **ReplayBuffer**: Stores and samples training examples
- **ChessLoss**: Implements the loss function for the neural network
- **ModelEvaluator**: Evaluates model performance by playing games

## Technical Design Decisions

### Neural Network Architecture

The neural network uses a deep residual architecture with:

- 19 input channels representing the board state
- Multiple residual blocks with batch normalization
- Two output heads: policy (move probabilities) and value (win probability)

This design is inspired by AlphaZero and has been proven effective for chess.

### Monte Carlo Tree Search

The MCTS implementation uses:

- Upper Confidence Bounds (UCB) for exploration/exploitation balance
- Virtual loss for parallelization
- Dirichlet noise at the root for exploration during self-play
- Temperature parameter for controlling move selection randomness

### Computer Vision Approach

The board detection uses a multi-stage process:

1. Adaptive thresholding to handle different lighting conditions
2. Contour detection to find the chess board
3. Perspective transformation to get a top-down view
4. Grid-based extraction of individual squares

Piece classification uses a convolutional neural network trained on thousands of chess piece images.

### Data Flow

Training data flows through the system as follows:

1. Self-play games generate (state, policy, value) tuples
2. Data is stored in the replay buffer with a FIFO policy
3. Training samples random batches for stochastic gradient descent
4. Trained models are evaluated against previous versions
5. Better models become the new baseline for self-play

## Scalability and Performance

The system is designed to scale in several dimensions:

- **Training**: Supports distributed self-play for faster training
- **Inference**: Can adjust MCTS simulations based on available compute
- **Vision**: Detection algorithms adapt to different screen resolutions and chess UIs

Performance optimizations include:

- Caching detected board coordinates between frames
- Batch processing squares for piece classification
- Parallel MCTS simulations where possible
- GPU acceleration for neural network inference

## Future Architecture Extensions

The modular design allows for future extensions:

- **Plugin system** for different chess platforms
- **API interface** for programmatic control
- **Distributed architecture** for cloud deployment
- **Reinforcement learning from human games** alongside self-play
