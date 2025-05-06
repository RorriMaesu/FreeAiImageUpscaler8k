#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration management for the AI Image Upscaler
"""

import os
import json
import logging
from pathlib import Path

from utils.error_handler import ConfigError, exception_handler
from utils.logger import get_logger

logger = get_logger()

class Config:
    """Configuration manager for the application"""

    DEFAULT_CONFIG = {
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
        "use_gpu": True
    }

    def __init__(self, config_path=None):
        """Initialize configuration"""
        self.config_path = config_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        self.config = self._load_config()

        # Ensure required directories exist
        self._ensure_directories()

    @exception_handler
    def _load_config(self):
        """Load configuration from file or create default"""
        logger.info(f"Loading configuration from {self.config_path}")
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with default config to ensure all keys exist
                merged_config = self.DEFAULT_CONFIG.copy()
                merged_config.update(config)
                logger.info("Configuration loaded successfully")
                return merged_config
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in config file: {e}")
                logger.warning("Using default configuration")
                return self.DEFAULT_CONFIG.copy()
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                logger.warning("Using default configuration")
                return self.DEFAULT_CONFIG.copy()
        else:
            # Create default config
            logger.info("Configuration file not found, creating default")
            self._save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()

    @exception_handler
    def _save_config(self, config):
        """Save configuration to file"""
        logger.debug(f"Saving configuration to {self.config_path}")
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.debug("Configuration saved successfully")
        except Exception as e:
            error_msg = f"Error saving config: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg, e)

    @exception_handler
    def _ensure_directories(self):
        """Ensure required directories exist"""
        logger.debug("Ensuring required directories exist")
        base_dir = os.path.dirname(os.path.dirname(__file__))

        try:
            # Models directory
            models_dir = os.path.join(base_dir, self.config["models_dir"])
            os.makedirs(models_dir, exist_ok=True)
            logger.debug(f"Models directory ensured: {models_dir}")

            # Output directory
            output_dir = os.path.join(base_dir, self.config["output_dir"])
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Output directory ensured: {output_dir}")

            # Logs directory
            logs_dir = os.path.join(base_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            logger.debug(f"Logs directory ensured: {logs_dir}")
        except Exception as e:
            error_msg = f"Error creating directories: {e}"
            logger.error(error_msg)
            raise ConfigError(error_msg, e)

    @exception_handler
    def get(self, key, default=None):
        """Get configuration value"""
        value = self.config.get(key, default)
        logger.debug(f"Config get: {key} = {value}")
        return value

    @exception_handler
    def set(self, key, value):
        """Set configuration value and save"""
        logger.info(f"Setting config: {key} = {value}")
        self.config[key] = value
        self._save_config(self.config)

    @exception_handler
    def get_model_info(self, model_name):
        """Get information about a specific model"""
        model_info = self.config["available_models"].get(model_name, None)
        if model_info is None:
            logger.warning(f"Model info not found for: {model_name}")
        else:
            logger.debug(f"Retrieved model info for: {model_name}")
        return model_info

    @exception_handler
    def get_models_dir(self):
        """Get absolute path to models directory"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        models_dir = os.path.join(base_dir, self.config["models_dir"])
        logger.debug(f"Models directory: {models_dir}")
        return models_dir

    @exception_handler
    def get_output_dir(self):
        """Get absolute path to output directory"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        output_dir = os.path.join(base_dir, self.config["output_dir"])
        logger.debug(f"Output directory: {output_dir}")
        return output_dir

    @exception_handler
    def get_logs_dir(self):
        """Get absolute path to logs directory"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        logs_dir = os.path.join(base_dir, "logs")
        logger.debug(f"Logs directory: {logs_dir}")
        return logs_dir
