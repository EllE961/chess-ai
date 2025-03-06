# Usage Instructions

This document provides comprehensive instructions for using the Chess AI system to play games on digital chess platforms.

## Quick Start

To start playing immediately with default settings:

```bash
python play.py --color white
```

This will:

1. Calibrate the system to detect the chess board
2. Start a new game as white on the configured chess platform
3. Play the game autonomously until completion

## Command-Line Interface

The system provides three main command-line scripts:

1. `play.py`: For playing games
2. `main.py`: For advanced operations and calibration
3. `train.py`: For training the neural network (see [Training Documentation](training.md))

### Playing Games

```bash
python play.py [options]
```

Options:

- `--config PATH`: Path to configuration file (default: `config/hyperparameters.yaml`)
- `--color {white,black,random}`: Color to play as (default: `white`)
- `--manual`: Don't automatically start a new game (use existing game)
- `--games N`: Number of games to play (default: 1)

Examples:

```bash
# Play as black
python play.py --color black

# Play 5 consecutive games with random colors
python play.py --color random --games 5

# Use an existing game on screen (no auto-start)
python play.py --manual
```

### Advanced Operations

```bash
python main.py [options]
```

Options:

- `--config PATH`: Path to configuration file
- `--mode {play,train,calibrate}`: Operation mode
- `--color {white,black,random}`: Color to play as (for play mode)

Examples:

```bash
# Just calibrate the system
python main.py --mode calibrate

# Advanced play mode with custom configuration
python main.py --mode play --config custom_config.yaml --color black
```

## Calibration

Calibration is a critical step that teaches the system how to recognize the chess board and map screen coordinates to chess squares.

### Automatic Calibration

Calibration is performed automatically when you run the play script. If you want to run calibration separately:

```bash
python main.py --mode calibrate
```

During calibration:

1. The system captures the screen
2. Detects the chess board using computer vision
3. Determines board orientation (white or black perspective)
4. Maps screen coordinates to chess squares
5. Saves the calibration data for future use

### Manual Calibration

If automatic calibration fails, you can use the interactive calibration tool:

```bash
python -c "
import sys; sys.path.append('.')
from automation.calibrator import Calibrator
from config.config import load_config
calib = Calibrator(load_config())
calib.interactive_calibration()
"
```

Follow the on-screen instructions to manually mark the corners of the chess board.

## Platform Configuration

The system can work with different chess platforms. The default is Lichess, but you can configure it for other platforms.

### Supported Platforms

- **Lichess** (`lichess`): Default platform
- **Chess.com** (`chess.com`): Popular commercial platform
- **Chess24** (`chess24`): Professional chess platform
- **Custom** (`custom`): Generic platform support

### Setting the Platform

Edit the `config/hyperparameters.yaml` file and change the `platform` parameter:

```yaml
automation:
  platform: "lichess" # Change to "chess.com", "chess24", or "custom"
```

### Platform Templates

The system uses template images to recognize UI elements on different platforms. To capture these templates:

```bash
python scripts/capture_templates.py --interactive
```

This will guide you through capturing images of:

- Platform logo
- "Your turn" indicator
- Game control buttons
- Other platform-specific elements

## Playing Modes

### 1. Autonomous Mode

The default mode where the AI plays completely autonomously:

```bash
python play.py --color white
```

The system will:

- Start a new game on the configured platform
- Play moves automatically when it's the AI's turn
- Continue until the game is finished

### 2. Manual Game Start

If you want to manually start a game and then let the AI take over:

```bash
python play.py --manual
```

In this mode:

1. You start a game on the chess platform manually
2. The AI detects the active game
3. The AI starts playing automatically when it's its turn

### 3. Continuous Play

To play multiple games in succession:

```bash
python play.py --games 5
```

The system will:

1. Play a complete game
2. Wait a short time after completion
3. Start a new game
4. Repeat until the specified number of games are played

## Performance Tuning

### Strength Settings

The playing strength of the AI depends on several configurable parameters:

- **MCTS Simulations**: More simulations yield stronger play but slower decisions

  ```yaml
  mcts:
    num_simulations: 800 # Increase for stronger play, decrease for faster moves
  ```

- **Exploration Constant**: Controls balance between exploration and exploitation

  ```yaml
  mcts:
    c_puct: 1.5 # Higher values (2-4) lead to more exploratory play
  ```

- **Temperature**: Controls randomness in move selection
  ```yaml
  mcts:
    temperature_init: 1.0 # Higher values (1-2) give more varied play
    temperature_final: 0.1 # Lower values (0-0.5) give more optimal play
  ```

### Speed Settings

To optimize speed for weaker hardware:

- **Reduce MCTS simulations**:

  ```yaml
  mcts:
    num_simulations: 200 # Faster but weaker play
  ```

- **Adjust polling interval**:

  ```yaml
  automation:
    polling_interval: 2.0 # Check game state less frequently (seconds)
  ```

- **Optimize move execution**:
  ```yaml
  automation:
    mouse_move_delay: 0.05 # Faster mouse movements
    move_execution_delay: 0.2 # Faster clicking
  ```

## Troubleshooting

### Common Issues

1. **Board Detection Fails**

   - Ensure the chess board is fully visible on screen
   - Try adjusting the board size or browser zoom level
   - Run manual calibration: `python main.py --mode calibrate`

2. **Incorrect Moves**

   - Recalibrate the system: `python main.py --mode calibrate`
   - Check if the board perspective is correctly detected
   - Verify that pieces are correctly classified

3. **System Too Slow**

   - Reduce MCTS simulations
   - Close other applications to free system resources
   - Use a GPU for neural network inference

4. **Crashes During Play**
   - Check logs in the `logs/game_logs` directory
   - Ensure all dependencies are correctly installed
   - Try running with debug mode enabled

### Enabling Debug Mode

For more detailed logging and visualizations:

1. Edit `config/hyperparameters.yaml`:

   ```yaml
   system:
     debug_mode: true
     log_level: "DEBUG"
   ```

2. This will:
   - Save board detection images to `logs/vision_debug`
   - Display move visualizations
   - Provide detailed logging information

### Viewing Logs

Logs are stored in the `logs` directory:

- `logs/game_logs`: Logs from game play
- `logs/training_logs`: Logs from training
- `logs/vision_debug`: Debug images from the vision system

## Advanced Usage

### Custom Configurations

You can create custom configuration files for different scenarios:

```bash
cp config/hyperparameters.yaml config/custom_config.yaml
# Edit custom_config.yaml
python play.py --config config/custom_config.yaml
```

### API Integration

For programmers, the system can be imported and used from Python code:

```python
from chess_ai.main import ChessAI

# Initialize the AI
chess_ai = ChessAI("config/hyperparameters.yaml")

# Calibrate
chess_ai.calibrate()

# Play a game
chess_ai.start_new_game(color="white")
```

This allows integration with other systems or custom user interfaces.

### Platform-Specific Tips

**Lichess**

- Works best with the default board theme
- Set "Piece animation" to "Normal" in preferences
- Use the standard board size

**Chess.com**

- Use the "Green" or "Tournament" board theme for best detection
- Disable piece animations in settings
- Keep the browser window in focus

**Chess24**

- Use the standard board theme
- Disable animation effects

## Next Steps

After mastering basic usage:

1. [Train your own model](training.md) to improve playing strength
2. Explore [system architecture](architecture.md) to understand the components
3. Consider [contributing](../CONTRIBUTING.md) to the project
