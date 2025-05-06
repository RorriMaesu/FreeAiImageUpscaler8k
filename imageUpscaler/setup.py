#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for AI Image Upscaler
"""

import os
import sys
import json
import traceback
import logging

def setup():
    """Set up the application directory structure"""
    try:
        print("Setting up AI Image Upscaler...")

        # Create required directories
        directories = [
            "models",
            "output",
            "logs",
            "temp"
        ]

        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✓ Created directory: {directory}")
            except Exception as e:
                print(f"✗ Error creating directory {directory}: {e}")
                raise

        # Create default config file if it doesn't exist
        config_path = "config.json"
        if not os.path.exists(config_path):
            try:
                default_config = {
                    "models_dir": "models",
                    "output_dir": "output",
                    "default_model": "realesrgan-x4plus",
                    "available_models": {
                        "realesrgan-x4plus": {
                            "name": "RealESRGAN x4plus",
                            "scale": 4,
                            "type": "photo",
                            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                        },
                        "realesrgan-x4plus-anime": {
                            "name": "RealESRGAN x4plus Anime",
                            "scale": 4,
                            "type": "anime",
                            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
                        },
                        "realesrgan-x2plus": {
                            "name": "RealESRGAN x2plus",
                            "scale": 2,
                            "type": "photo",
                            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
                        },
                        "swinir-large": {
                            "name": "SwinIR Large",
                            "scale": 4,
                            "type": "photo",
                            "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
                        }
                    },
                    "tile_size": 512,
                    "tile_padding": 32,
                    "face_enhance": False,
                    "use_gpu": True,
                    "log_level": "INFO"
                }

                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)

                print(f"✓ Created default configuration file: {config_path}")
            except Exception as e:
                print(f"✗ Error creating configuration file: {e}")
                raise
        else:
            print(f"✓ Configuration file already exists: {config_path}")

            # Update existing config file with new fields if needed
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Check for missing fields and add them
                updated = False
                if "log_level" not in config:
                    config["log_level"] = "INFO"
                    updated = True

                # Save updated config if changes were made
                if updated:
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=4)
                    print(f"✓ Updated configuration file with new fields")
            except Exception as e:
                print(f"✗ Error updating configuration file: {e}")
                # Don't raise here, just continue with setup

        # Create a default logging configuration
        logging_config_path = os.path.join("logs", "logging.conf")
        if not os.path.exists(logging_config_path):
            try:
                logging_config = """[loggers]
keys=root

[handlers]
keys=consoleHandler,fileHandler

[formatters]
keys=simpleFormatter,detailedFormatter

[logger_root]
level=INFO
handlers=consoleHandler,fileHandler

[handler_consoleHandler]
class=StreamHandler
level=INFO
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=handlers.RotatingFileHandler
level=DEBUG
formatter=detailedFormatter
args=('logs/upscaler.log', 'a', 10485760, 5)

[formatter_simpleFormatter]
format=%(levelname)s: %(message)s

[formatter_detailedFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
"""
                # Create logs directory if it doesn't exist
                os.makedirs(os.path.dirname(logging_config_path), exist_ok=True)

                with open(logging_config_path, 'w') as f:
                    f.write(logging_config)

                print(f"✓ Created logging configuration: {logging_config_path}")
            except Exception as e:
                print(f"✗ Error creating logging configuration: {e}")
                # Don't raise here, just continue with setup

        print("\nSetup complete! You can now run the application with 'python main.py'")
        return True

    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        print("\nError details:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = setup()
    sys.exit(0 if success else 1)
