#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the installation of AI Image Upscaler
"""

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

def test_installation():
    """Test if all required dependencies are installed correctly"""
    print("Testing AI Image Upscaler installation...")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # Check PyTorch
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Error checking PyTorch: {e}")
    
    # Check OpenCV
    try:
        print(f"OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"Error checking OpenCV: {e}")
    
    # Check NumPy
    try:
        print(f"NumPy version: {np.__version__}")
    except Exception as e:
        print(f"Error checking NumPy: {e}")
    
    # Check PIL
    try:
        print(f"PIL version: {Image.__version__}")
    except Exception as e:
        print(f"Error checking PIL: {e}")
    
    # Check directory structure
    print("\nChecking directory structure:")
    required_dirs = ["models", "output"]
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✓ {directory} directory exists")
        else:
            print(f"✗ {directory} directory not found")
    
    # Check configuration file
    if os.path.exists("config.json"):
        print("✓ Configuration file exists")
    else:
        print("✗ Configuration file not found")
    
    print("\nInstallation test complete!")

if __name__ == "__main__":
    test_installation()
