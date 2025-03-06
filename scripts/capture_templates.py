#!/usr/bin/env python
"""
Script to capture template images for platform detection.

This script helps to capture and save template images of chess platform UI elements
that will be used for platform detection and game state analysis.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import pyautogui
from PIL import Image, ImageGrab

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config
from utils.logger import setup_logger

def capture_template(name, x, y, width, height, output_dir):
    """
    Capture a region of the screen and save it as a template.
    
    Args:
        name: Name of the template
        x, y: Top-left coordinates of the region
        width, height: Dimensions of the region
        output_dir: Directory to save the template
    """
    # Capture the region
    region = (x, y, width, height)
    screenshot = ImageGrab.grab(bbox=region)
    
    # Convert to numpy array
    template = np.array(screenshot)
    
    # Save the template
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.png")
    cv2.imwrite(output_path, cv2.cvtColor(template, cv2.COLOR_RGB2BGR))
    
    print(f"Template '{name}' saved to {output_path}")
    return template

def capture_with_mouse():
    """
    Interactive template capture using mouse selection.
    
    The user selects a region by clicking and dragging the mouse.
    """
    print("Click and drag to select a region. Press Esc to cancel.")
    
    # Use pyautogui to capture mouse events
    start_x, start_y = pyautogui.position()
    print(f"Starting point: ({start_x}, {start_y})")
    
    # Wait for mouse release
    while pyautogui.mouseDown():
        time.sleep(0.1)
    
    end_x, end_y = pyautogui.position()
    print(f"Ending point: ({end_x}, {end_y})")
    
    # Calculate region
    x = min(start_x, end_x)
    y = min(start_y, end_y)
    width = abs(end_x - start_x)
    height = abs(end_y - start_y)
    
    return x, y, width, height

def capture_templates_interactive(config_path):
    """
    Interactive script to capture templates for all platforms.
    
    Args:
        config_path: Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    template_dir = config.get('template_dir', './templates')
    
    # Set up logging
    logger = setup_logger(config)
    
    # Define the templates to capture
    templates = [
        {"name": "lichess_logo", "platform": "Lichess", "description": "Lichess logo (usually in the top-left corner)"},
        {"name": "lichess_your_turn", "platform": "Lichess", "description": "Your turn indicator on Lichess"},
        {"name": "lichess_game_controls", "platform": "Lichess", "description": "Game control buttons on Lichess"},
        {"name": "chess_com_logo", "platform": "Chess.com", "description": "Chess.com logo"},
        {"name": "chess_com_your_turn", "platform": "Chess.com", "description": "Your turn indicator on Chess.com"},
        {"name": "chess_com_game_controls", "platform": "Chess.com", "description": "Game control buttons on Chess.com"},
        {"name": "chess24_logo", "platform": "Chess24", "description": "Chess24 logo"},
        {"name": "chess24_your_turn", "platform": "Chess24", "description": "Your turn indicator on Chess24"},
        {"name": "chess24_game_controls", "platform": "Chess24", "description": "Game control buttons on Chess24"}
    ]
    
    print("Chess AI Template Capture Tool")
    print("==============================")
    print("This tool will help you capture template images for platform detection.")
    print("Please open the chess platforms in your browser and prepare to select UI elements.")
    
    # Capture each template
    for template in templates:
        print(f"\nCapturing template: {template['name']}")
        print(f"Platform: {template['platform']}")
        print(f"Description: {template['description']}")
        
        input("Press Enter when ready to select the region (then click and drag) or Ctrl+C to skip...")
        
        try:
            # Let the user select a region
            x, y, width, height = capture_with_mouse()
            
            # Capture the template
            capture_template(template['name'], x, y, width, height, template_dir)
            
        except KeyboardInterrupt:
            print(f"Skipping {template['name']}")
            continue
    
    print("\nTemplate capture complete!")
    print(f"Templates are saved in: {template_dir}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Capture template images for platform detection")
    parser.add_argument("--config", type=str, default="config/hyperparameters.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--name", type=str, help="Name of the template to capture")
    parser.add_argument("--output-dir", type=str, help="Directory to save templates")
    parser.add_argument("--interactive", action="store_true", help="Interactive template capture")
    
    args = parser.parse_args()
    
    if args.interactive:
        capture_templates_interactive(args.config)
    elif args.name:
        # Manual template capture
        x, y, width, height = capture_with_mouse()
        output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
        capture_template(args.name, x, y, width, height, output_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()