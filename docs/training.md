# Training Documentation

This document provides detailed instructions and best practices for training the neural network model used by the Chess AI system.

## Training Overview

The Chess AI uses a reinforcement learning approach similar to AlphaZero, where the model improves through self-play without any human gameplay data. The training process consists of the following steps:

1. **Self-play**: The AI plays games against itself using MCTS guided by the current neural network
2. **Data collection**: Game states, moves, and outcomes are collected as training examples
3. **Neural network training**: The network is trained to predict:
   - Move probabilities (policy) that match the improved MCTS search
   - Game outcomes (value) for each position
4. **Model evaluation**: The new model is tested against the previous best model
5. **Model selection**: If the new model performs better, it becomes the new best model
6. **Iteration**: The process is repeated with the improved model generating better self-play data

## Quick Start

To start training with default settings:

```bash
python train.py
```

For more control:

```bash
python train.py --config custom_config.yaml --iterations 50 --resume
```

Or use the training script:

```bash
./scripts/train.sh --iterations 100 --resume
```

## Training Parameters

The main training parameters are defined in `config/hyperparameters.yaml`. Here are the key parameters you might want to adjust:

### Self-Play Settings

- `num_self_play_games`: Number of games to play in each iteration (default: 25)
- `num_simulations`: Number of MCTS simulations per move (default: 800)
- `temperature_init`: Initial temperature for move exploration (default: 1.0)
- `temperature_final`: Final temperature after threshold (default: 0.1)
- `move_temp_threshold`: Move number to switch to final temperature (default: 30)

### Neural Network Training

- `batch_size`: Training batch size (default: 2048)
- `learning_rate`: Initial learning rate (default: 0.01)
- `weight_decay`: L2 regularization parameter (default: 0.0001)
- `lr_step_size`: Number of iterations before learning rate decay (default: 20)
- `lr_gamma`: Learning rate decay factor (default: 0.1)
- `epochs_per_iteration`: Training epochs per iteration (default: 10)

### Neural Network Architecture

- `num_res_blocks`: Number of residual blocks in the network (default: 19)
- `num_filters`: Number of filters in convolutional layers (default: 256)

## Hardware Requirements

Training is computationally intensive. Here are the recommended hardware specifications:

- **CPU**: 8+ cores for faster self-play
- **RAM**: 16GB+ for larger batch sizes
- **GPU**: 8GB+ VRAM for faster training
- **Storage**: 10GB+ for model checkpoints and training data

GPU acceleration is highly recommended, as it can speed up training by 10-50x compared to CPU-only training.

## Training Phases

For optimal results, we recommend training in phases:

### Phase 1: Initial Training (Iterations 1-20)

- Focus: Learn basic chess rules and simple tactics
- Use lower MCTS simulations (200-400) for faster iterations
- Higher learning rate (0.01-0.02)
- Small to medium batch size (1024-2048)

Example command:

```bash
python train.py --iterations 20
```

### Phase 2: Tactical Development (Iterations 21-50)

- Focus: Develop tactical understanding
- Increase MCTS simulations (600-800)
- Medium learning rate (0.001-0.01)
- Medium batch size (2048-4096)

Example command:

```bash
python train.py --resume --iterations 30
```

### Phase 3: Strategic Refinement (Iterations 51+)

- Focus: Develop strategic understanding
- High MCTS simulations (800-1600)
- Low learning rate (0.0001-0.001)
- Large batch size (4096-8192)

Example command:

```bash
python train.py --resume --iterations 50
```

## Monitoring Training Progress

Training progress is logged to the console and saved to log files in the `logs/training_logs` directory. Additionally, TensorBoard logs are created for visualization:

```bash
tensorboard --logdir logs
```

Key metrics to monitor:

- **Loss values**: Total loss, policy loss, and value loss should decrease over time
- **Accuracy**: Policy and value accuracy should increase
- **Performance**: Win rate against previous model versions should improve
- **Learning rate**: Should decrease according to the schedule

## Saving and Loading Models

Models are automatically saved during training:

- `best_model.pt`: The best performing model so far
- `model_iter_X.pt`: Model after iteration X
- `current_model.pt`: The most recent model

To continue training from a saved model:

```bash
python train.py --resume
```

To start from a specific iteration:

```bash
python train.py --resume_iteration 25
```

## Training Tips

1. **Start Small**: Begin with smaller networks and fewer MCTS simulations to iterate quickly
2. **Gradually Increase**: As training progresses, increase network size and MCTS simulations
3. **Monitor Overfitting**: If validation loss increases while training loss decreases, adjust regularization
4. **Save Checkpoints**: Regularly save models to avoid losing progress
5. **Use Multiple GPUs**: If available, distribute self-play across multiple GPUs for faster training
6. **Experiment**: Try different hyperparameters to find what works best for your hardware

## Evaluating Models

To evaluate how well your model plays, you can:

1. **Compare against previous versions**:

   ```bash
   ./scripts/evaluate.sh --model1 models/best_model.pt --model2 models/model_iter_25.pt --games 40
   ```

2. **Play against it manually**:

   ```bash
   python play.py --color black
   ```

3. **Analyze self-play games** in the `data/game_records` directory

## Troubleshooting

### Common Training Issues

1. **Out of Memory Errors**

   - Reduce batch size
   - Reduce network size
   - Reduce number of self-play games stored in memory

2. **Slow Training Progress**

   - Increase learning rate temporarily
   - Check GPU utilization (should be >90%)
   - Optimize data loading with more workers

3. **Model Not Improving**

   - Increase exploration (higher temperature/Dirichlet noise)
   - Adjust MCTS parameters
   - Verify training data quality

4. **Oscillating Performance**
   - Reduce learning rate
   - Increase evaluation games for more reliable comparisons
   - Implement a moving average for model selection

## Advanced Training

For advanced users, the system supports:

- **Distributed training** across multiple machines
- **Custom neural network architectures** by modifying `core/neural_network.py`
- **Custom MCTS implementations** by modifying `core/mcts.py`
- **External opening books** for more diverse training games

Refer to the source code and comments for more details on these advanced features.
