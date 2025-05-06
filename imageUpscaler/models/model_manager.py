#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model manager for AI Image Upscaler
"""

import os
import torch
import traceback
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from utils.downloader import ModelDownloader
from utils.error_handler import ModelError, exception_handler
from utils.logger import get_logger

logger = get_logger()

class ModelManager:
    """Manager for upscaling models"""

    @exception_handler
    def __init__(self, config):
        """Initialize model manager with configuration"""
        self.config = config
        self.models_dir = config.get_models_dir()
        logger.debug(f"Models directory: {self.models_dir}")

        # Initialize downloader
        self.downloader = ModelDownloader(self.models_dir)

        # Set device
        use_gpu = config.get('use_gpu', True)
        cuda_available = torch.cuda.is_available()

        if use_gpu and not cuda_available:
            logger.warning("GPU requested but CUDA is not available. Using CPU instead.")

        self.device = torch.device('cuda' if cuda_available and use_gpu else 'cpu')
        logger.info(f"Model manager using device: {self.device}")

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            logger.info(f"GPU: {gpu_name} with {gpu_memory:.2f} GB memory")

        self.loaded_model = None
        self.current_model_name = None

    @exception_handler
    def get_available_models(self):
        """Get list of available models"""
        models = self.config.get("available_models", {})
        logger.debug(f"Available models: {list(models.keys())}")
        return models

    @exception_handler
    def download_model(self, model_name, callback=None):
        """Download a model if not already available"""
        logger.info(f"Requesting download for model: {model_name}")

        # Get model info
        model_info = self.config.get_model_info(model_name)
        if not model_info:
            error_msg = f"Unknown model: {model_name}"
            logger.error(error_msg)
            raise ModelError(error_msg)

        # Get model URL
        model_url = model_info.get("url")
        if not model_url:
            error_msg = f"No URL specified for model: {model_name}"
            logger.error(error_msg)
            raise ModelError(error_msg)

        # Download model
        try:
            model_path = self.downloader.download_model(model_name, model_url, callback)
            logger.info(f"Model downloaded successfully: {model_path}")
            return model_path
        except Exception as e:
            error_msg = f"Failed to download model {model_name}: {e}"
            logger.error(error_msg)
            raise ModelError(error_msg, e)

    @exception_handler
    def load_model(self, model_name):
        """Load a model for inference"""
        logger.info(f"Loading model: {model_name}")

        # Check if model is already loaded
        if self.loaded_model is not None and self.current_model_name == model_name:
            logger.info(f"Model {model_name} is already loaded")
            return self.loaded_model

        # Unload current model if different
        if self.loaded_model is not None:
            logger.debug(f"Unloading current model: {self.current_model_name}")
            self.unload_model()

        # Get model info
        model_info = self.config.get_model_info(model_name)
        if not model_info:
            error_msg = f"Unknown model: {model_name}"
            logger.error(error_msg)
            raise ModelError(error_msg)

        # Check if model file exists, download if needed
        model_path = self.downloader.get_model_path(model_name)
        if not os.path.exists(model_path):
            logger.info(f"Model file not found, downloading: {model_name}")
            model_path = self.download_model(model_name)

        # Load model based on type
        try:
            if "realesrgan" in model_name.lower():
                logger.debug(f"Loading RealESRGAN model: {model_name}")
                model = self._load_realesrgan_model(model_name, model_path, model_info)
            elif "swinir" in model_name.lower():
                logger.debug(f"Loading SwinIR model: {model_name}")
                model = self._load_swinir_model(model_name, model_path, model_info)
            else:
                error_msg = f"Unsupported model type: {model_name}"
                logger.error(error_msg)
                raise ModelError(error_msg)

            # Store loaded model
            self.loaded_model = model
            self.current_model_name = model_name

            # Log model details
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model {model_name} loaded successfully ({num_params/1e6:.2f}M parameters)")

            return model
        except Exception as e:
            error_msg = f"Error loading model {model_name}: {e}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ModelError(error_msg, e)

    @exception_handler
    def _load_realesrgan_model(self, model_name, model_path, model_info):
        """Load a RealESRGAN model"""
        logger.debug(f"Creating RealESRGAN architecture for {model_name}")

        try:
            # Get scale factor
            scale = model_info.get("scale", 4)
            logger.debug(f"Model scale factor: {scale}x")

            # Create model architecture
            if "anime" in model_name:
                logger.debug("Using anime-optimized architecture (6 blocks)")
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=scale)
            else:
                logger.debug("Using standard architecture (23 blocks)")
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)

            # Load model weights
            logger.debug(f"Loading weights from {model_path}")
            loadnet = torch.load(model_path, map_location=self.device)

            # Determine key name for weights
            if 'params_ema' in loadnet:
                keyname = 'params_ema'
                logger.debug("Using EMA parameters")
            else:
                keyname = 'params'
                logger.debug("Using standard parameters")

            # Load state dict
            model.load_state_dict(loadnet[keyname], strict=True)

            # Set to evaluation mode
            model.eval()

            # Move to device
            logger.debug(f"Moving model to device: {self.device}")
            model = model.to(self.device)

            return model
        except Exception as e:
            error_msg = f"Error loading RealESRGAN model {model_name}: {e}"
            logger.error(error_msg)
            raise ModelError(error_msg, e)

    @exception_handler
    def _load_swinir_model(self, model_name, model_path, model_info):
        """Load a SwinIR model"""
        logger.debug(f"Creating SwinIR architecture for {model_name}")

        try:
            # Import SwinIR model
            try:
                from models.swinir_model import SwinIR
                logger.debug("SwinIR model imported successfully")
            except ImportError as e:
                error_msg = "SwinIR model requires additional dependencies. Please install them first."
                logger.error(error_msg)
                raise ModelError(error_msg, e)

            # Get scale factor
            scale = model_info.get("scale", 4)
            logger.debug(f"Model scale factor: {scale}x")

            # Create model architecture
            logger.debug("Creating SwinIR model with default parameters")
            model = SwinIR(
                upscale=scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="1conv"
            )

            # Load model weights
            logger.debug(f"Loading weights from {model_path}")
            loadnet = torch.load(model_path, map_location=self.device)

            # Load state dict
            model.load_state_dict(loadnet['params'], strict=True)

            # Set to evaluation mode
            model.eval()

            # Move to device
            logger.debug(f"Moving model to device: {self.device}")
            model = model.to(self.device)

            return model
        except Exception as e:
            if not isinstance(e, ModelError):  # Avoid double-wrapping
                error_msg = f"Error loading SwinIR model {model_name}: {e}"
                logger.error(error_msg)
                raise ModelError(error_msg, e)
            raise

    @exception_handler
    def unload_model(self):
        """Unload the current model to free memory"""
        if self.loaded_model is not None:
            logger.info(f"Unloading model: {self.current_model_name}")

            try:
                # Delete model
                del self.loaded_model

                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    logger.debug("Clearing CUDA cache")
                    torch.cuda.empty_cache()

                # Reset model references
                self.loaded_model = None
                self.current_model_name = None

                logger.debug("Model unloaded successfully")
            except Exception as e:
                error_msg = f"Error unloading model: {e}"
                logger.error(error_msg)
                raise ModelError(error_msg, e)
