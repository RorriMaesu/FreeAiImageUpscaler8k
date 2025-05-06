#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify error handling and logging in AI Image Upscaler
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Initialize logger first
from utils.logger import Logger
logger = Logger().get_logger()

# Import other modules
from utils.config import Config
from utils.error_handler import (
    UpscalerError, ModelError, ImageProcessingError, ConfigError, 
    NetworkError, log_exception
)
from utils.downloader import ModelDownloader
from models.model_manager import ModelManager
from utils.image_processor import ImageProcessor

def test_config_errors():
    """Test error handling in Config class"""
    logger.info("Testing Config error handling...")
    
    try:
        # Test with invalid config path
        invalid_path = "nonexistent/path/config.json"
        logger.info(f"Testing with invalid config path: {invalid_path}")
        config = Config(invalid_path)
        logger.info("Config created successfully with default values")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    
    try:
        # Test with invalid key
        invalid_key = "nonexistent_key"
        logger.info(f"Testing with invalid config key: {invalid_key}")
        value = config.get(invalid_key)
        logger.info(f"Got default value for invalid key: {value}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    
    logger.info("Config error handling tests passed")
    return True

def test_downloader_errors():
    """Test error handling in ModelDownloader class"""
    logger.info("Testing ModelDownloader error handling...")
    
    # Create temporary directory for testing
    test_dir = "test_models"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Create downloader
        downloader = ModelDownloader(test_dir)
        
        # Test with invalid URL
        model_name = "test_model"
        invalid_url = "https://invalid.url/model.pth"
        
        logger.info(f"Testing with invalid URL: {invalid_url}")
        try:
            downloader.download_model(model_name, invalid_url)
            logger.error("Download should have failed but didn't")
            return False
        except Exception as e:
            logger.info(f"Download failed as expected: {e}")
        
        # Test with valid model check
        exists = downloader.check_model_exists(model_name)
        logger.info(f"Model exists check: {exists}")
        
        # Get model path
        model_path = downloader.get_model_path(model_name)
        logger.info(f"Model path: {model_path}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    finally:
        # Clean up test directory
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                file_path = os.path.join(test_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(test_dir)
    
    logger.info("ModelDownloader error handling tests passed")
    return True

def test_image_processor_errors():
    """Test error handling in ImageProcessor class"""
    logger.info("Testing ImageProcessor error handling...")
    
    config = Config()
    processor = ImageProcessor(config)
    
    try:
        # Test with nonexistent image
        nonexistent_image = "nonexistent_image.jpg"
        logger.info(f"Testing with nonexistent image: {nonexistent_image}")
        
        try:
            processor.load_image(nonexistent_image)
            logger.error("Image loading should have failed but didn't")
            return False
        except Exception as e:
            logger.info(f"Image loading failed as expected: {e}")
        
        # Test with invalid output path
        invalid_output_path = "/invalid/path/output.jpg"
        logger.info(f"Testing with invalid output path: {invalid_output_path}")
        
        try:
            # Create a dummy image
            import numpy as np
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            processor.save_image(dummy_image, invalid_output_path)
            logger.error("Image saving should have failed but didn't")
            return False
        except Exception as e:
            logger.info(f"Image saving failed as expected: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    
    logger.info("ImageProcessor error handling tests passed")
    return True

def test_model_manager_errors():
    """Test error handling in ModelManager class"""
    logger.info("Testing ModelManager error handling...")
    
    config = Config()
    manager = ModelManager(config)
    
    try:
        # Test with nonexistent model
        nonexistent_model = "nonexistent_model"
        logger.info(f"Testing with nonexistent model: {nonexistent_model}")
        
        try:
            manager.load_model(nonexistent_model)
            logger.error("Model loading should have failed but didn't")
            return False
        except Exception as e:
            logger.info(f"Model loading failed as expected: {e}")
        
        # Test with valid model but force download error
        valid_model = "realesrgan-x4plus"
        logger.info(f"Testing with valid model but forcing download error: {valid_model}")
        
        # Temporarily modify the downloader to force an error
        original_download = manager.downloader.download_model
        
        def mock_download(*args, **kwargs):
            raise NetworkError("Simulated network error")
        
        try:
            manager.downloader.download_model = mock_download
            manager.download_model(valid_model)
            logger.error("Download should have failed but didn't")
            return False
        except Exception as e:
            logger.info(f"Download failed as expected: {e}")
        finally:
            # Restore original method
            manager.downloader.download_model = original_download
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    
    logger.info("ModelManager error handling tests passed")
    return True

def run_tests():
    """Run all error handling tests"""
    logger.info("Starting error handling tests...")
    
    tests = [
        test_config_errors,
        test_downloader_errors,
        test_image_processor_errors,
        test_model_manager_errors
    ]
    
    results = []
    for test_func in tests:
        try:
            logger.info(f"Running test: {test_func.__name__}")
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
            logger.error(traceback.format_exc())
            results.append((test_func.__name__, False))
    
    # Print summary
    logger.info("\nTest Results Summary:")
    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\nAll error handling tests passed!")
    else:
        logger.error("\nSome error handling tests failed!")
    
    return all_passed

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
