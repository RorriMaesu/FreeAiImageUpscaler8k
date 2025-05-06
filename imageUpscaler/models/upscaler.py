#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Upscaler module for AI Image Upscaler
"""

import os
import torch
import time
from PIL import Image
import numpy as np

from utils.image_processor import ImageProcessor
from models.model_manager import ModelManager
from utils.error_handler import UpscalerError, exception_handler
from utils.logger import get_logger

logger = get_logger()

class Upscaler:
    """Main upscaler class that handles the upscaling process"""

    @exception_handler
    def __init__(self, config):
        """Initialize upscaler with configuration"""
        logger.info("Initializing Upscaler")
        self.config = config

        # Initialize components
        logger.debug("Creating ModelManager")
        self.model_manager = ModelManager(config)

        logger.debug("Creating ImageProcessor")
        self.image_processor = ImageProcessor(config)

        # Set device
        use_gpu = config.get('use_gpu', True)
        cuda_available = torch.cuda.is_available()

        if use_gpu and not cuda_available:
            logger.warning("GPU requested but CUDA is not available. Using CPU instead.")

        self.device = torch.device('cuda' if cuda_available and use_gpu else 'cpu')
        logger.info(f"Upscaler using device: {self.device}")

        logger.info("Upscaler initialized successfully")

    @exception_handler
    def upscale_image(self, image_path, output_path=None, model_name=None, progress_callback=None):
        """
        Upscale an image using the specified model

        Args:
            image_path: Path to the input image
            output_path: Path to save the output image (optional)
            model_name: Name of the model to use (optional)
            progress_callback: Callback function to report progress (optional)

        Returns:
            Path to the upscaled image
        """
        start_time = time.time()
        logger.info(f"Starting upscaling for image: {image_path}")

        try:
            # Use default model if not specified
            if model_name is None:
                model_name = self.config.get('default_model')
                logger.debug(f"Using default model: {model_name}")

            # Get model info
            logger.debug(f"Getting info for model: {model_name}")
            model_info = self.config.get_model_info(model_name)
            if not model_info:
                error_msg = f"Unknown model: {model_name}"
                logger.error(error_msg)
                raise UpscalerError(error_msg)

            # Load model
            logger.info(f"Loading model: {model_name}")
            model = self.model_manager.load_model(model_name)

            # Load image
            logger.info(f"Loading image: {image_path}")
            img = self.image_processor.load_image(image_path)

            # Report progress
            if progress_callback:
                logger.debug("Reporting progress: 10%")
                progress_callback(10)  # 10% - Image loaded

            # Get scale factor
            scale = model_info.get('scale', 4)
            logger.info(f"Upscaling with factor: {scale}x")

            # Process image with tiling
            logger.info("Converting image to numpy array")
            img_np = np.array(img)

            # Create progress callback wrapper
            def progress_wrapper(progress):
                if progress_callback:
                    adjusted_progress = 10 + int(progress * 0.8)
                    logger.debug(f"Reporting progress: {adjusted_progress}%")
                    progress_callback(adjusted_progress)

            logger.info("Starting image processing with tiling")
            output = self.image_processor.process_image_with_tiling(
                model, img_np, scale, callback=progress_wrapper
            )

            # Generate output path if not specified
            if output_path is None:
                filename, ext = os.path.splitext(os.path.basename(image_path))
                output_dir = self.config.get_output_dir()
                output_path = os.path.join(output_dir, f"{filename}_upscaled{ext}")
                logger.debug(f"Generated output path: {output_path}")

            # Save output image
            logger.info(f"Saving output image to: {output_path}")
            self.image_processor.save_image(output, output_path)

            # Report progress
            if progress_callback:
                logger.debug("Reporting progress: 100%")
                progress_callback(100)  # 100% - Complete

            # Calculate processing time
            elapsed_time = time.time() - start_time
            logger.info(f"Upscaling completed in {elapsed_time:.2f} seconds")

            return output_path

        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Error upscaling image {image_path}: {e}"
            logger.error(error_msg)
            logger.error(f"Upscaling failed after {elapsed_time:.2f} seconds")
            raise UpscalerError(error_msg, e)

    @exception_handler
    def upscale_batch(self, image_paths, output_dir=None, model_name=None, progress_callback=None):
        """
        Upscale multiple images using the specified model

        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save output images (optional)
            model_name: Name of the model to use (optional)
            progress_callback: Callback function to report progress (optional)

        Returns:
            List of paths to upscaled images
        """
        start_time = time.time()
        logger.info(f"Starting batch upscaling for {len(image_paths)} images")

        try:
            # Use default model if not specified
            if model_name is None:
                model_name = self.config.get('default_model')
                logger.debug(f"Using default model: {model_name}")

            # Use default output directory if not specified
            if output_dir is None:
                output_dir = self.config.get_output_dir()
                logger.debug(f"Using default output directory: {output_dir}")

            # Create output directory if it doesn't exist
            logger.debug(f"Ensuring output directory exists: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

            # Process each image
            output_paths = []
            total_images = len(image_paths)
            successful_images = 0
            failed_images = 0

            logger.info(f"Processing {total_images} images")

            for i, image_path in enumerate(image_paths):
                # Generate output path
                filename, ext = os.path.splitext(os.path.basename(image_path))
                output_path = os.path.join(output_dir, f"{filename}_upscaled{ext}")

                # Report batch progress
                if progress_callback:
                    batch_progress = int(i / total_images * 100)
                    logger.debug(f"Reporting batch progress: {batch_progress}% ({i+1}/{total_images})")
                    progress_callback(batch_progress, i, total_images)

                # Upscale image
                try:
                    logger.info(f"Processing image {i+1}/{total_images}: {image_path}")

                    # Create a dummy progress callback to avoid progress reporting
                    # for individual images in batch mode
                    def dummy_callback(_):
                        pass

                    output_path = self.upscale_image(
                        image_path,
                        output_path,
                        model_name,
                        dummy_callback
                    )
                    output_paths.append(output_path)
                    successful_images += 1
                    logger.info(f"Successfully processed image: {image_path}")
                except Exception as e:
                    failed_images += 1
                    error_msg = f"Error upscaling image {image_path}: {e}"
                    logger.error(error_msg)
                    # Continue with next image

            # Report final progress
            if progress_callback:
                logger.debug("Reporting final batch progress: 100%")
                progress_callback(100, total_images, total_images)

            # Calculate processing time
            elapsed_time = time.time() - start_time
            avg_time_per_image = elapsed_time / total_images if total_images > 0 else 0

            logger.info(f"Batch processing completed in {elapsed_time:.2f} seconds")
            logger.info(f"Average time per image: {avg_time_per_image:.2f} seconds")
            logger.info(f"Successfully processed {successful_images}/{total_images} images")

            if failed_images > 0:
                logger.warning(f"Failed to process {failed_images}/{total_images} images")

            return output_paths

        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Error in batch upscaling: {e}"
            logger.error(error_msg)
            logger.error(f"Batch processing failed after {elapsed_time:.2f} seconds")
            raise UpscalerError(error_msg, e)

    @exception_handler
    def get_available_models(self):
        """Get list of available models"""
        logger.debug("Getting list of available models")
        models = self.model_manager.get_available_models()
        logger.debug(f"Found {len(models)} available models")
        return models

    @exception_handler
    def download_model(self, model_name, callback=None):
        """Download a model if not already available"""
        logger.info(f"Requesting download for model: {model_name}")

        try:
            model_path = self.model_manager.download_model(model_name, callback)
            logger.info(f"Model downloaded successfully: {model_path}")
            return model_path
        except Exception as e:
            error_msg = f"Error downloading model {model_name}: {e}"
            logger.error(error_msg)
            raise UpscalerError(error_msg, e)
