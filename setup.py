"""
Setup script for the chess AI package.

This script allows the package to be installed using pip.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chess-ai",
    version="0.1.0",
    author="EllE",
    author_email="yahyaalsalmi961@gmail.com",
    description="Autonomous Chess AI System with Reinforcement Learning and Computer Vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EllE961/chess-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "opencv-python>=4.4.0",
        "pillow>=8.0.0",
        "python-chess>=1.3.0",
        "pyautogui>=0.9.50",
        "pynput>=1.7.0",
        "matplotlib>=3.3.0",
        "pyyaml>=5.3.0",
        "tqdm>=4.46.0",
        "pytest>=6.0.0",
        "tensorboard>=2.4.0",
        "cairosvg>=2.5.0",
    ],
    entry_points={
        "console_scripts": [
            "chess-ai=main:main",
            "chess-ai-train=train:main",
            "chess-ai-play=play:main",
        ],
    },
)