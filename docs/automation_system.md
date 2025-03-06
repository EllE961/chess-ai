# Automation System Documentation

This document provides a detailed explanation of the automation system used to control the mouse and interact with chess platforms.

## System Overview

The automation system is responsible for:

1. Translating chess moves to mouse actions
2. Interacting with different chess platform UIs
3. Calibrating screen coordinates to chess squares
4. Detecting game state changes

## Components

The automation system consists of three main components:

1. **MoveExecutor**: Controls mouse movements and clicks to execute chess moves
2. **PlatformAdapter**: Handles platform-specific UI elements and interactions
3. **Calibrator**: Calibrates screen coordinates for accurate mouse control

## Move Execution

### Mouse Control Architecture

The move executor uses a layered approach to mouse control:

1. **High-level API**: `execute_move(move)` - Takes a chess move and handles the execution
2. **Mid-level API**: `_execute_mouse_move(from_coords, to_coords)` - Handles the mouse movement sequence
3. **Low-level API**: `_move_mouse_humanlike(coords)` - Controls the actual mouse movement

This layered approach makes it easy to modify or extend the behavior at any level.

### Human-like Movement

The system uses several techniques to create human-like mouse movements:

1. **Path Randomization**: The mouse follows a slightly curved path rather than moving in a straight line
2. **Variable Speed**: The mouse accelerates and decelerates naturally
3. **Position Randomness**: The exact click position has a small random offset
4. **Timing Variation**: Delays between actions vary slightly

These features make the automation harder to detect as a bot by chess platforms.

### Configuration Parameters

Key parameters that control move execution behavior:

```yaml
automation:
  mouse_move_delay: 0.1 # Base delay between mouse movements
  move_execution_delay: 0.5 # Delay between click actions
  click_randomness: 5 # Maximum random offset for clicks (pixels)
```

### Special Move Handling

The move executor has special handling for:

1. **Promotions**: Detects promotion moves and clicks the appropriate piece type
2. **Castling**: Executes castling as a single move (king to new position)
3. **En Passant**: Handles capture correctly by clicking captured square

### Alternative Movement Methods

The system supports two mouse control strategies:

1. **Click-Click**: Click the source square, then click the destination square
2. **Drag-and-Drop**: Press at the source, drag to destination, release

The default is click-click, but some platforms work better with drag-and-drop.

## Platform Adaptation

### Supported Platforms

The system currently supports three major chess platforms:

1. **Lichess**: Open-source chess server with a clean UI
2. **Chess.com**: Commercial chess platform with more UI elements
3. **Chess24**: Professional chess platform with advanced features

A generic/custom platform mode is also available for other chess UIs.

### Platform-Specific Adaptations

Each platform requires specific handling:

| Platform  | UI Navigation                  | Game State Detection       | Move Execution              |
| --------- | ------------------------------ | -------------------------- | --------------------------- |
| Lichess   | Minimalist UI, easy navigation | Uses "Your turn" indicator | Click-click works well      |
| Chess.com | Complex UI with animations     | Uses clock highlighting    | May need drag-and-drop      |
| Chess24   | Professional UI                | Uses move list updates     | Special timing requirements |

### Template Matching

The platform adapter uses template matching to identify UI elements:

1. **Template Images**: The system stores reference images for UI elements (logos, buttons, indicators)
2. **Template Matching**: OpenCV's template matching algorithm finds these elements on screen
3. **Confidence Threshold**: A match score above the threshold identifies an element

### Game State Detection

The platform adapter detects the game state using multiple methods:

1. **UI Element Detection**: Looks for elements like "Your turn" indicators
2. **Visual Change Detection**: Detects when the board changes (opponent moved)
3. **Board Orientation**: Determines if we're playing as white or black
4. **Game Controls**: Detects when game controls are visible (game in progress)

### Starting New Games

The system can automatically start new games:

1. **Navigation**: Clicks through the platform's UI to start a game
2. **Color Selection**: Can select to play as white, black, or random
3. **Game Mode**: Configurable game mode and time control
4. **Opponent Finding**: Waits for an opponent to be found

## Calibration

### Calibration Process

The calibration process maps screen coordinates to chess squares:

1. **Board Detection**: Find the chess board on screen using computer vision
2. **Corner Extraction**: Extract the four corners of the board
3. **Perspective Analysis**: Determine if the board is from white's or black's perspective
4. **Coordinate Mapping**: Create a mapping from chess squares (0-63) to screen coordinates
5. **Validation**: Test the mapping by highlighting each square

### Interactive Calibration

For difficult cases, the system provides interactive calibration:

1. **Automatic Attempt**: First tries automatic board detection
2. **Manual Correction**: If automatic detection fails, allows manual corner marking
3. **Coordinate Visualization**: Shows the mapped squares on screen for confirmation
4. **Fine-tuning**: Allows adjustments to the board corners

### Coordinate Transformation

The system uses perspective transformation for accurate coordinate mapping:

1. **Bilinear Interpolation**: Maps chess coordinates (file, rank) to screen coordinates (x, y)
2. **Perspective Handling**: Accounts for board perspective (white/black) when mapping
3. **Edge Handling**: Special handling for edge squares where UI elements might interfere

## Workflow Integration

The three components work together in the following workflow:

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Platform      │────>│ Calibrator    │────>│ Move Executor │
│ Adapter       │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
        │                                            │
        │                                            │
        ▼                                            ▼
┌───────────────┐                           ┌───────────────┐
│ Game State    │                           │ Mouse Actions │
│ Detection     │                           │               │
└───────────────┘                           └───────────────┘
```

1. The **Platform Adapter** detects which chess platform is being used
2. The **Calibrator** maps the chess board to screen coordinates
3. The **Move Executor** translates chess moves to mouse actions
4. The **Platform Adapter** continuously monitors game state

## Performance Considerations

### Timing and Reliability

Timing is critical for reliable automation:

1. **Detection Intervals**: Game state is checked periodically (default: 1.0 seconds)
2. **Move Execution Timing**: Delays between mouse actions help ensure reliable execution
3. **Platform Response Time**: Different platforms have different response times

### Resource Usage

The automation system is designed to be lightweight:

- **CPU**: Minimal usage for mouse control
- **Memory**: Very low footprint
- **GPU**: Not used for automation components

### Error Handling

The system includes robust error handling:

1. **Move Verification**: Verifies that moves were successfully executed
2. **Retry Logic**: Automatically retries failed moves
3. **Failure Recovery**: Can continue playing even after some failures

## Platform Extension

### Adding New Platforms

To add support for a new chess platform:

1. Create template images for the platform's UI elements
2. Extend the `PlatformAdapter` class with platform-specific methods
3. Implement platform-specific game state detection
4. Test and tune the timing parameters

### Custom UI Handling

For unusual chess UIs, you can:

1. Create a custom platform adapter
2. Implement specialized detection methods
3. Override the default mouse control behavior

## Debugging and Testing

### Debug Visualization

The automation system provides debugging visualizations:

1. **Board Mapping**: Visualizes the chess square to screen coordinate mapping
2. **Mouse Path**: Shows the planned mouse movement path
3. **UI Detection**: Highlights detected UI elements

### Testing Utilities

Several utilities are available for testing:

1. **Test Mode**: Simulates moves without actually moving the mouse
2. **Step Mode**: Executes moves one step at a time
3. **Recording**: Records all automation actions for review

## Common Issues and Solutions

### Reliability Issues

Common reliability issues and their solutions:

1. **Inconsistent Detection**

   - Adjust the detection thresholds
   - Create new templates for your specific platform theme
   - Increase detection frequency

2. **Missed Clicks**

   - Increase move execution delay
   - Adjust the click randomness
   - Try the drag-and-drop method instead

3. **Platform Changes**
   - Update the template images after platform UI updates
   - Adjust timing parameters for new platform behavior
   - Update platform-specific logic

### Security Considerations

Important security notes:

1. **Terms of Service**: Be aware that some chess platforms may prohibit automation
2. **Rate Limiting**: Avoid excessive move speeds that could trigger anti-bot measures
3. **Human-like Behavior**: Use human-like timing and movement patterns

## Advanced Features

### Game Management

The system can manage entire gaming sessions:

1. **Multiple Games**: Play sequential games automatically
2. **Time Management**: Adjust play speed based on remaining time
3. **Result Tracking**: Track and log game results

### UI Interaction

Beyond move execution, the system can:

1. **Chat Interaction**: Respond to chat messages with predefined texts
2. **Settings Adjustment**: Change game settings between games
3. **Event Handling**: Handle popups, notifications, and other interruptions

## Troubleshooting Guide

### Mouse Control Issues

1. **Mouse Moves to Wrong Location**

   - Recalibrate the board coordinates
   - Check if the board was moved or resized
   - Verify that the perspective detection is correct

2. **Clicks Not Registering**

   - Increase the click duration
   - Check if the platform requires drag-and-drop
   - Verify that the window has focus

3. **Promotion Not Working**
   - Adjust the promotion piece offset
   - Try with different timing values
   - Check if the platform uses a non-standard promotion UI

### Platform Detection Issues

1. **Platform Not Detected**

   - Update template images
   - Try with a standard theme/UI setting
   - Check if the platform has been updated recently

2. **Game State Incorrectly Detected**
   - Adjust detection thresholds
   - Add more specific UI element templates
   - Implement a custom detection method for your platform
