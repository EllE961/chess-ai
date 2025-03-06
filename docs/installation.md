# Installation Guide

This guide provides detailed instructions for installing the Chess AI system and its dependencies.

## System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+ recommended)
- **Python**: Python 3.8 or higher
- **Hardware**:
  - CPU: Multi-core processor (4+ cores recommended)
  - RAM: Minimum 8GB (16GB+ recommended for training)
  - GPU: CUDA-compatible NVIDIA GPU (optional but recommended for training)
  - Storage: 1GB+ free disk space
  - Screen: 1920x1080 or higher resolution

## Installation Methods

There are three ways to install the Chess AI system:

1. [Automated Installation](#automated-installation) (Recommended)
2. [Manual Installation](#manual-installation)
3. [Development Installation](#development-installation)

## Automated Installation

The automated installation script handles all setup steps for you.

### On Linux/macOS:

```bash
# Clone the repository
git clone https://github.com/EllE961/chess-ai.git
cd chess-ai

# Make the setup script executable
chmod +x scripts/setup.sh

# Run the setup script
./scripts/setup.sh
```

### On Windows:

```powershell
# Clone the repository
git clone https://github.com/EllE961/chess-ai.git
cd chess-ai

# Run the setup script
python scripts\setup_windows.py
```

## Manual Installation

If you prefer to install manually or the automated script doesn't work for your environment:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/EllE961/chess-ai.git
   cd chess-ai
   ```

2. **Create and activate a virtual environment**:

   - On Linux/macOS:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Create necessary directories**:

   ```bash
   mkdir -p models data/game_records logs/training_logs logs/game_logs templates
   ```

5. **Install the package**:
   ```bash
   pip install -e .
   ```

## Development Installation

For development work, you'll need additional tools and dependencies:

1. **Follow the manual installation steps 1-3** above.

2. **Install development dependencies**:

   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Set up pre-commit hooks**:

   ```bash
   pre-commit install
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

## GPU Support

To enable GPU acceleration (highly recommended for training):

1. **Install CUDA and cuDNN**:

   - Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Download and install [cuDNN](https://developer.nvidia.com/cudnn)
   - Follow NVIDIA's installation instructions for your platform

2. **Install GPU version of PyTorch**:

   ```bash
   pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
   ```

   (Replace version numbers as appropriate for your CUDA version)

3. **Verify GPU setup**:

   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
   ```

   This should print `True` followed by the number of available GPUs.

## Platform-Specific Notes

### Windows

- You may need to install Microsoft Visual C++ Build Tools for some packages
- Some packages may require admin privileges to install

### macOS

- You may need to install Xcode Command Line Tools: `xcode-select --install`
- GPU acceleration is not available on macOS for PyTorch

### Linux

- You may need to install additional system packages:
  ```bash
  sudo apt-get update
  sudo apt-get install -y python3-dev python3-pip python3-venv python3-opencv
  sudo apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
  ```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'cv2'**

   - Solution: Reinstall OpenCV with `pip install opencv-python`

2. **GPU not detected**

   - Check CUDA installation: `nvcc --version`
   - Verify GPU is recognized: `nvidia-smi`
   - Make sure PyTorch CUDA version matches your CUDA Toolkit version

3. **Mouse control not working**

   - On Linux, you may need to install additional packages:
     ```bash
     sudo apt-get install python3-tk python3-dev
     ```

4. **Permission errors**
   - Use `sudo` for system-level installations
   - Make sure you have write permissions to the installation directory

For more troubleshooting help, refer to the [FAQ](faq.md) or [create an issue](https://github.com/EllE961/chess-ai/issues) on GitHub.

## Next Steps

After installation:

1. [Calibrate the system](usage.md#calibration)
2. [Train the neural network](training.md)
3. [Play your first game](usage.md#playing-games)
