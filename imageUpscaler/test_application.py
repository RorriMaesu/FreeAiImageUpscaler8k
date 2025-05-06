#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive test script for AI Image Upscaler
"""

import os
import sys
import time
import argparse
import traceback
import numpy as np
from PIL import Image
import torch

# Initialize logger first
from utils.logger import Logger
logger = Logger().get_logger()

# Import other modules
from utils.config import Config
from utils.error_handler import UpscalerError
from models.upscaler import Upscaler

def create_test_image(output_path, width=256, height=256):
    """Create a test image for upscaling"""
    logger.info(f"Creating test image: {output_path}")
    
    try:
        # Create a gradient test image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add horizontal gradient (red channel)
        for x in range(width):
            img[:, x, 0] = int(255 * x / width)
        
        # Add vertical gradient (green channel)
        for y in range(height):
            img[y, :, 1] = int(255 * y / height)
        
        # Add diagonal pattern (blue channel)
        for y in range(height):
            for x in range(width):
                img[y, x, 2] = int(255 * (x + y) / (width + height))
        
        # Add some shapes for detail testing
        # Circle
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        for y in range(height):
            for x in range(width):
                if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                    img[y, x] = [255, 255, 255]
        
        # Save the image
        Image.fromarray(img).save(output_path)
        logger.info(f"Test image created successfully: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error creating test image: {e}")
        raise

def test_upscaling(model_name=None, test_image_path=None):
    """Test the upscaling process"""
    logger.info("Testing upscaling process")
    
    try:
        # Create configuration
        config = Config()
        
        # Create upscaler
        upscaler = Upscaler(config)
        
        # Use default model if not specified
        if model_name is None:
            model_name = config.get('default_model')
        
        logger.info(f"Using model: {model_name}")
        
        # Create test image if not provided
        if test_image_path is None:
            test_dir = "temp"
            os.makedirs(test_dir, exist_ok=True)
            test_image_path = os.path.join(test_dir, "test_image.png")
            create_test_image(test_image_path)
        
        # Progress callback
        def progress_callback(progress):
            logger.info(f"Upscaling progress: {progress}%")
        
        # Upscale the image
        logger.info(f"Starting upscaling of {test_image_path}")
        start_time = time.time()
        
        output_path = upscaler.upscale_image(
            test_image_path,
            model_name=model_name,
            progress_callback=progress_callback
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Upscaling completed in {elapsed_time:.2f} seconds")
        logger.info(f"Output saved to: {output_path}")
        
        # Verify the output image exists and is valid
        if not os.path.exists(output_path):
            logger.error(f"Output file does not exist: {output_path}")
            return False
        
        try:
            output_img = Image.open(output_path)
            width, height = output_img.size
            logger.info(f"Output image dimensions: {width}x{height}")
            
            # Get model info to verify scale factor
            model_info = config.get_model_info(model_name)
            scale = model_info.get('scale', 4)
            
            # Get input image dimensions
            input_img = Image.open(test_image_path)
            input_width, input_height = input_img.size
            
            # Verify dimensions match expected scale
            expected_width = input_width * scale
            expected_height = input_height * scale
            
            if width != expected_width or height != expected_height:
                logger.error(f"Output dimensions {width}x{height} don't match expected {expected_width}x{expected_height}")
                return False
            
            logger.info(f"Output dimensions verified: {width}x{height} (scale: {scale}x)")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying output image: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error in upscaling test: {e}")
        logger.error(traceback.format_exc())
        return False

def test_batch_upscaling(model_name=None, num_images=3):
    """Test batch upscaling process"""
    logger.info(f"Testing batch upscaling with {num_images} images")
    
    try:
        # Create configuration
        config = Config()
        
        # Create upscaler
        upscaler = Upscaler(config)
        
        # Use default model if not specified
        if model_name is None:
            model_name = config.get('default_model')
        
        logger.info(f"Using model: {model_name}")
        
        # Create test images
        test_dir = "temp"
        os.makedirs(test_dir, exist_ok=True)
        
        test_image_paths = []
        for i in range(num_images):
            test_image_path = os.path.join(test_dir, f"test_image_{i}.png")
            create_test_image(test_image_path, 
                             width=128 + i * 32,  # Vary sizes
                             height=128 + i * 32)
            test_image_paths.append(test_image_path)
        
        # Progress callback
        def progress_callback(progress, current, total):
            logger.info(f"Batch progress: {progress}% ({current}/{total})")
        
        # Upscale the images
        logger.info(f"Starting batch upscaling of {len(test_image_paths)} images")
        start_time = time.time()
        
        output_paths = upscaler.upscale_batch(
            test_image_paths,
            model_name=model_name,
            progress_callback=progress_callback
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Batch upscaling completed in {elapsed_time:.2f} seconds")
        logger.info(f"Average time per image: {elapsed_time/len(test_image_paths):.2f} seconds")
        
        # Verify all output images exist
        success = True
        for output_path in output_paths:
            if not os.path.exists(output_path):
                logger.error(f"Output file does not exist: {output_path}")
                success = False
        
        if success:
            logger.info(f"All {len(output_paths)} output images verified")
        
        return success
            
    except Exception as e:
        logger.error(f"Error in batch upscaling test: {e}")
        logger.error(traceback.format_exc())
        return False

def test_model_download(model_name=None):
    """Test model downloading"""
    logger.info("Testing model download")
    
    try:
        # Create configuration
        config = Config()
        
        # Create upscaler
        upscaler = Upscaler(config)
        
        # Use default model if not specified
        if model_name is None:
            model_name = config.get('default_model')
        
        logger.info(f"Testing download for model: {model_name}")
        
        # Progress callback
        def progress_callback(model_name, progress):
            logger.info(f"Download progress for {model_name}: {progress}%")
        
        # Download the model
        start_time = time.time()
        
        model_path = upscaler.download_model(model_name, progress_callback)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Download completed in {elapsed_time:.2f} seconds")
        logger.info(f"Model saved to: {model_path}")
        
        # Verify the model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file does not exist: {model_path}")
            return False
        
        logger.info(f"Model file verified: {model_path}")
        return True
            
    except Exception as e:
        logger.error(f"Error in model download test: {e}")
        logger.error(traceback.format_exc())
        return False

def run_tests(args):
    """Run all tests"""
    logger.info("Starting comprehensive application tests")
    
    tests = []
    
    # Add tests based on arguments
    if args.all or args.download:
        tests.append(("Model Download", lambda: test_model_download(args.model)))
    
    if args.all or args.upscale:
        tests.append(("Single Image Upscaling", lambda: test_upscaling(args.model, args.image)))
    
    if args.all or args.batch:
        tests.append(("Batch Upscaling", lambda: test_batch_upscaling(args.model, args.num_images)))
    
    # Run tests
    results = []
    for name, test_func in tests:
        logger.info(f"\n=== Running test: {name} ===")
        try:
            start_time = time.time()
            result = test_func()
            elapsed_time = time.time() - start_time
            
            status = "PASSED" if result else "FAILED"
            logger.info(f"Test {name}: {status} (Time: {elapsed_time:.2f}s)")
            results.append((name, result, elapsed_time))
        except Exception as e:
            logger.error(f"Test {name} failed with exception: {e}")
            logger.error(traceback.format_exc())
            results.append((name, False, 0))
    
    # Print summary
    logger.info("\n=== Test Results Summary ===")
    all_passed = True
    total_time = 0
    
    for name, result, elapsed_time in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{name}: {status} (Time: {elapsed_time:.2f}s)")
        if not result:
            all_passed = False
        total_time += elapsed_time
    
    logger.info(f"\nTotal test time: {total_time:.2f} seconds")
    
    if all_passed:
        logger.info("\nAll tests passed!")
    else:
        logger.error("\nSome tests failed!")
    
    return all_passed

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test AI Image Upscaler")
    
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--download", action="store_true", help="Test model download")
    parser.add_argument("--upscale", action="store_true", help="Test image upscaling")
    parser.add_argument("--batch", action="store_true", help="Test batch upscaling")
    parser.add_argument("--model", type=str, help="Model to use for testing")
    parser.add_argument("--image", type=str, help="Image to use for testing")
    parser.add_argument("--num-images", type=int, default=3, help="Number of test images for batch testing")
    
    args = parser.parse_args()
    
    # If no test specified, run all tests
    if not (args.all or args.download or args.upscale or args.batch):
        args.all = True
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    success = run_tests(args)
    sys.exit(0 if success else 1)
