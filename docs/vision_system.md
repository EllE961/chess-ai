# Vision System Documentation

This document provides detailed information about the computer vision system used to detect chess boards and pieces on the screen.

## System Overview

The vision system is responsible for:

1. Detecting the chess board on the screen
2. Extracting individual squares from the board
3. Classifying the pieces on each square
4. Converting the visual representation to a FEN string

## Components

The vision system consists of three main components:

1. **BoardDetector**: Detects the chess board and its coordinates on the screen
2. **PieceClassifier**: Identifies the chess pieces on each square
3. **PositionExtractor**: Converts the classified pieces to a FEN string

## Board Detection

### Detection Algorithm

The board detection uses the following approach:

1. **Capture Screen**: Take a screenshot of the current screen
2. **Pre-processing**:
   - Convert to grayscale
   - Apply adaptive thresholding
3. **Contour Detection**:
   - Find contours in the binary image
   - Filter contours by size and shape
4. **Quadrilateral Detection**:
   - Find contours that are approximately quadrilateral
   - Check that internal angles are close to 90 degrees
5. **Corner Extraction**:
   - Extract the four corners of the board
   - Order them as top-left, top-right, bottom-right, bottom-left
6. **Perspective Transformation**:
   - Apply perspective transformation to get a bird's-eye view
   - Create a square warped image of the board

### Fallback Methods

If the primary detection method fails, the system uses fallback approaches:

1. **Hough Line Detection**:

   - Detect straight lines using the Hough transform
   - Find intersections to identify the board grid
   - Extract the outermost intersections as corners

2. **Previous Detection**:
   - Use the previously detected board coordinates
   - Useful when only minor changes occur between frames

### Configuration Parameters

Key parameters that affect board detection:

```yaml
vision:
  board_detection_threshold: 0.8 # Threshold for contour filtering
  detection_interval: 0.5 # Time between detection attempts (seconds)
```

## Piece Classification

### Classification Approach

Piece classification uses a convolutional neural network (CNN) with the following architecture:

1. **Input**: 64x64 RGB image of a chess square
2. **Convolutional Layers**:
   - 3 conv blocks with increasing filter counts (32, 64, 128)
   - Each block has conv2d → batch normalization → ReLU → max pooling
3. **Fully Connected Layers**:
   - Flatten → 512 units → Dropout → 13 output classes
4. **Output Classes**:
   - Empty square
   - White pieces (pawn, knight, bishop, rook, queen, king)
   - Black pieces (pawn, knight, bishop, rook, queen, king)

### Training the Classifier

The piece classifier is trained on a dataset of chess piece images:

1. **Data Collection**:

   - Screenshots of various chess boards
   - Different piece styles and themes
   - Different lighting conditions

2. **Training Process**:

   - Data augmentation (rotation, flipping, contrast adjustments)
   - Cross-entropy loss function
   - Adam optimizer
   - Early stopping based on validation accuracy

3. **Performance Metrics**:
   - Classification accuracy >95% on validation set
   - Confusion matrix to identify problematic piece types

### Confidence Threshold

The system uses a confidence threshold to filter out uncertain predictions:

```yaml
vision:
  min_piece_confidence: 0.7 # Minimum confidence for piece classification
```

If confidence is below this threshold, the square is classified as empty.

## Position Extraction

### FEN Generation

The position extractor converts the classified pieces to a FEN string:

1. **Board Orientation**:

   - Determine if the board is viewed from white's or black's perspective
   - Flip the board if necessary

2. **FEN Construction**:

   - Create the piece placement part of the FEN
   - Add the active color, castling rights, en passant target, and move counters
   - Use context from previous positions when available

3. **Validation & Correction**:
   - Verify that the generated FEN is valid
   - Apply heuristic corrections if invalid
   - Find the most similar valid position

### FEN Validation

The validator checks several aspects of the position:

1. **Syntax Check**: Verify FEN format is correct
2. **Piece Count**: Ensure reasonable numbers of each piece
3. **King Check**: Exactly one king per side
4. **Pawn Check**: No pawns on first/last ranks

## Workflow Integration

The three components work together in the following workflow:

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ Screen Capture│────>│ Board Detector│────>│Piece Classifier│────>│ Position      │
│               │     │               │     │               │     │ Extractor     │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
                             │                                            │
                             │                                            │
                             ▼                                            ▼
                      ┌───────────────┐                           ┌───────────────┐
                      │ Warped Board  │                           │   FEN String  │
                      │  Image        │                           │               │
                      └───────────────┘                           └───────────────┘
```

## Performance Considerations

### Optimization Techniques

The vision system employs several optimizations:

1. **Caching**: Board coordinates are cached between frames
2. **Periodic Detection**: Full detection runs only periodically
3. **Batch Processing**: All squares are classified in a single batch
4. **GPU Acceleration**: Neural network inference uses GPU when available

### Resource Usage

Approximate resource requirements:

- **CPU**: 10-20% utilization during active detection
- **RAM**: 100-200MB for vision components
- **GPU Memory**: 200-500MB when using GPU acceleration

### Latency

Typical latency measurements:

- **Board Detection**: 50-200ms
- **Piece Classification**: 30-150ms (GPU: 10-50ms)
- **Position Extraction**: 5-20ms
- **Total Pipeline**: 100-350ms (GPU: 70-250ms)

## Debugging and Visualization

The vision system provides debugging tools:

1. **Debug Images**:

   - Board detection visualization
   - Warped board images
   - Extracted squares grid
   - Classification confidence heatmap

2. **Enabling Debug Mode**:

   ```yaml
   system:
     debug_mode: true
   ```

3. **Debug Output Location**:
   - Debug images are saved to `logs/vision_debug/`
   - Timestamps in filenames for correlation with log entries

## Limitations and Edge Cases

Current limitations of the vision system:

1. **Lighting Conditions**:

   - Very dark or bright conditions may affect detection
   - Strong glare can interfere with piece classification

2. **Non-Standard Boards**:

   - Unusual board colors or patterns may not be detected
   - 3D piece sets can be difficult to classify accurately

3. **Overlapping Elements**:

   - UI elements overlapping the board can disrupt detection
   - Mouse cursor over pieces can affect classification

4. **Minimal Board Size**:
   - Board should be at least 400x400 pixels for reliable detection
   - Smaller boards may result in classification errors

## Improving the Vision System

### Training with Custom Piece Sets

To improve recognition of specific piece styles:

1. Capture screenshots of the specific piece set
2. Create a labeled dataset of squares
3. Train or fine-tune the piece classifier:
   ```bash
   python -m vision.piece_classifier --train --data custom_pieces_dataset
   ```

### Adjusting Detection Parameters

For difficult board styles, adjust these parameters:

```yaml
vision:
  board_detection_threshold: 0.7 # Lower for more lenient detection
  min_piece_confidence: 0.6 # Lower for more aggressive piece detection
```

### Adding New Board Themes

To add support for new board themes:

1. Create template images for the new theme
2. Place them in the `templates` directory
3. Update the platform adapter to use the new templates

## Troubleshooting

Common vision issues and solutions:

1. **Board Not Detected**

   - Make sure the board is completely visible on screen
   - Try with standard board colors (green and white/cream)
   - Adjust browser zoom level to change board size

2. **Pieces Misclassified**

   - Use a standard piece set (Alpha or Merida)
   - Avoid piece sets with unusual designs
   - Retrain the classifier with examples from your preferred set

3. **Incorrect Board Orientation**
   - Make sure the board corners are clearly visible
   - Use manual calibration to specify board corners
   - Check if the platform uses non-standard board orientation
