#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image processing utilities for AI Image Upscaler
"""

import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torchvision.transforms.functional import to_pil_image

from utils.error_handler import ImageProcessingError, exception_handler
from utils.logger import get_logger

logger = get_logger()

class ImageProcessor:
    """Utility for processing images"""

    @exception_handler
    def __init__(self, config):
        """Initialize processor with configuration"""
        self.config = config
        use_gpu = config.get('use_gpu', True)

        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        if use_gpu and not cuda_available:
            logger.warning("GPU requested but CUDA is not available. Using CPU instead.")

        # Set device
        self.device = torch.device('cuda' if cuda_available and use_gpu else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Get tiling parameters
        self.tile_size = config.get('tile_size', 512)
        self.tile_padding = config.get('tile_padding', 32)
        logger.debug(f"Tile size: {self.tile_size}, Tile padding: {self.tile_padding}")

    @exception_handler
    def load_image(self, image_path):
        """Load an image from file"""
        logger.info(f"Loading image: {image_path}")

        if not os.path.exists(image_path):
            error_msg = f"Image file not found: {image_path}"
            logger.error(error_msg)
            raise ImageProcessingError(error_msg)

        try:
            img = Image.open(image_path)

            # Check if image is valid
            img.verify()

            # Reopen after verify (which closes the file)
            img = Image.open(image_path).convert('RGB')

            # Get image info
            width, height = img.size
            logger.debug(f"Image loaded: {image_path} ({width}x{height})")

            return img
        except UnidentifiedImageError as e:
            error_msg = f"Unidentified image format: {image_path}"
            logger.error(error_msg)
            raise ImageProcessingError(error_msg, e)
        except Exception as e:
            error_msg = f"Error loading image {image_path}: {e}"
            logger.error(error_msg)
            raise ImageProcessingError(error_msg, e)

    @exception_handler
    def save_image(self, img, output_path):
        """Save an image to file"""
        logger.info(f"Saving image to: {output_path}")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logger.debug(f"Ensured output directory exists: {os.path.dirname(output_path)}")

            # Convert image to appropriate format
            if isinstance(img, torch.Tensor):
                logger.debug("Converting tensor to PIL image")
                img = to_pil_image(img)

            if isinstance(img, np.ndarray):
                logger.debug(f"Converting numpy array to PIL image (shape: {img.shape}, dtype: {img.dtype})")
                if img.dtype == np.float32:
                    img = (img * 255.0).astype(np.uint8)
                img = Image.fromarray(img)

            # Save the image
            img.save(output_path)
            logger.debug(f"Image saved successfully: {output_path}")

            return output_path
        except Exception as e:
            error_msg = f"Error saving image {output_path}: {e}"
            logger.error(error_msg)
            raise ImageProcessingError(error_msg, e)

    @exception_handler
    def preprocess_image(self, img):
        """Preprocess image for model input"""
        logger.debug("Preprocessing image for model input")

        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(img, Image.Image):
                logger.debug("Converting PIL Image to numpy array")
                img = np.array(img)

            # Input is already in RGB format when loaded from PIL
            # No color conversion needed
            logger.debug("Using RGB format directly")

            # Convert to float32 and normalize
            logger.debug("Normalizing image")
            img = img.astype(np.float32) / 255.0

            # Convert to tensor and add batch dimension
            logger.debug("Converting to tensor")
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

            # Move to device
            logger.debug(f"Moving tensor to device: {self.device}")
            return tensor.to(self.device)
        except Exception as e:
            error_msg = f"Error preprocessing image: {e}"
            logger.error(error_msg)
            raise ImageProcessingError(error_msg, e)

    @exception_handler
    def postprocess_image(self, tensor):
        """Postprocess model output to image"""
        logger.debug("Postprocessing model output")

        try:
            # Remove batch dimension and convert to numpy
            logger.debug("Converting tensor to numpy array")
            img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Clip values to [0, 1] range
            logger.debug("Clipping values to [0, 1] range")
            img = np.clip(img, 0, 1)

            # Convert to uint8
            logger.debug("Converting to uint8")
            img = (img * 255.0).astype(np.uint8)

            # PyTorch model outputs in RGB format, so no conversion needed
            logger.debug("Using RGB format directly")

            return img
        except Exception as e:
            error_msg = f"Error postprocessing image: {e}"
            logger.error(error_msg)
            raise ImageProcessingError(error_msg, e)

    @exception_handler
    def process_image_with_tiling(self, model, img, scale, callback=None):
        """
        Process an image using tiling to handle large images

        Args:
            model: The upscaling model
            img: Input image (PIL Image or numpy array)
            scale: Upscaling factor
            callback: Optional callback function to report progress

        Returns:
            Upscaled image as numpy array
        """
        logger.info(f"Processing image with tiling (scale: {scale}x)")

        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(img, Image.Image):
                logger.debug("Converting PIL Image to numpy array")
                img = np.array(img)

            # Get image dimensions
            h, w, c = img.shape
            logger.info(f"Input image dimensions: {w}x{h}x{c}")

            # Calculate output dimensions
            output_h, output_w = h * scale, w * scale
            logger.info(f"Output image dimensions: {output_w}x{output_h}x{c}")

            # Create output array
            logger.debug("Creating output array")
            output = np.zeros((output_h, output_w, c), dtype=np.uint8)

            # Calculate number of tiles
            num_tiles_x = (w + self.tile_size - 1) // self.tile_size
            num_tiles_y = (h + self.tile_size - 1) // self.tile_size
            total_tiles = num_tiles_x * num_tiles_y
            logger.info(f"Processing {total_tiles} tiles ({num_tiles_x}x{num_tiles_y})")

            # Check if we have enough memory for the operation
            estimated_memory_mb = (output_h * output_w * c * 4) / (1024 * 1024)  # float32 = 4 bytes
            logger.debug(f"Estimated output memory usage: {estimated_memory_mb:.2f} MB")

            tiles_processed = 0

            # Process each tile
            for y in range(0, h, self.tile_size):
                for x in range(0, w, self.tile_size):
                    # Calculate tile boundaries with padding
                    tile_x1 = max(0, x - self.tile_padding)
                    tile_y1 = max(0, y - self.tile_padding)
                    tile_x2 = min(w, x + self.tile_size + self.tile_padding)
                    tile_y2 = min(h, y + self.tile_size + self.tile_padding)

                    # Log tile information
                    logger.debug(f"Processing tile at ({x}, {y}), boundaries: ({tile_x1}, {tile_y1}) to ({tile_x2}, {tile_y2})")

                    try:
                        # Extract tile
                        tile = img[tile_y1:tile_y2, tile_x1:tile_x2, :]

                        # Preprocess tile
                        tile_tensor = self.preprocess_image(tile)

                        # Process tile with model
                        logger.debug("Running model inference on tile")
                        with torch.no_grad():
                            output_tensor = model(tile_tensor)

                        # Postprocess tile
                        output_tile = self.postprocess_image(output_tensor)

                        # Calculate the input tile dimensions
                        input_tile_h, input_tile_w = tile_y2 - tile_y1, tile_x2 - tile_x1

                        # Calculate the non-padded region in the input tile
                        non_pad_x1 = max(0, x - tile_x1)
                        non_pad_y1 = max(0, y - tile_y1)
                        non_pad_x2 = min(input_tile_w, non_pad_x1 + self.tile_size)
                        non_pad_y2 = min(input_tile_h, non_pad_y1 + self.tile_size)

                        # Calculate the corresponding region in the output tile
                        out_tile_x1 = non_pad_x1 * scale
                        out_tile_y1 = non_pad_y1 * scale
                        out_tile_x2 = non_pad_x2 * scale
                        out_tile_y2 = non_pad_y2 * scale

                        # Calculate the position in the final output image
                        out_x1 = x * scale
                        out_y1 = y * scale
                        out_x2 = min(output_w, out_x1 + self.tile_size * scale)
                        out_y2 = min(output_h, out_y1 + self.tile_size * scale)

                        logger.debug(f"Placing tile in output at ({out_x1}, {out_y1}) to ({out_x2}, {out_y2})")

                        # Extract the relevant part of the output tile
                        tile_part = output_tile[out_tile_y1:out_tile_y2, out_tile_x1:out_tile_x2]

                        # Place tile in output
                        output[out_y1:out_y2, out_x1:out_x2] = tile_part

                    except Exception as e:
                        # Log error but continue with other tiles
                        logger.error(f"Error processing tile at ({x}, {y}): {e}")
                        # If this is the first tile, re-raise the exception
                        if tiles_processed == 0:
                            raise

                    # Update progress
                    tiles_processed += 1
                    if callback:
                        progress = int(tiles_processed / total_tiles * 100)
                        logger.debug(f"Progress: {progress}% ({tiles_processed}/{total_tiles} tiles)")
                        callback(progress)

            logger.info("Image processing with tiling completed successfully")
            return output

        except Exception as e:
            error_msg = f"Error processing image with tiling: {e}"
            logger.error(error_msg)
            raise ImageProcessingError(error_msg, e)
