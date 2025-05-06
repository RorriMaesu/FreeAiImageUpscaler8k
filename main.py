#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI Image Upscaler - Launcher Script
"""

import os
import sys

def main():
    """Launch the AI Image Upscaler application"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the script directory to the Python path
    sys.path.insert(0, script_dir)
    
    # Import and run the main application
    from imageUpscaler.main import main as run_app
    return run_app()

if __name__ == "__main__":
    sys.exit(main())
