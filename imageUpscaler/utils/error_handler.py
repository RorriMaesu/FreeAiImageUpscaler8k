#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Error handling utility for AI Image Upscaler
"""

import sys
import traceback
from functools import wraps

from utils.logger import get_logger

logger = get_logger()

class UpscalerError(Exception):
    """Base exception class for AI Image Upscaler"""
    def __init__(self, message, original_error=None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)
    
    def __str__(self):
        if self.original_error:
            return f"{self.message} (Original error: {str(self.original_error)})"
        return self.message

class ModelError(UpscalerError):
    """Exception raised for errors related to models"""
    pass

class ImageProcessingError(UpscalerError):
    """Exception raised for errors related to image processing"""
    pass

class ConfigError(UpscalerError):
    """Exception raised for errors related to configuration"""
    pass

class NetworkError(UpscalerError):
    """Exception raised for errors related to network operations"""
    pass

def log_exception(e, context=""):
    """Log an exception with detailed information"""
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    tb_text = ''.join(tb_lines)
    
    if context:
        logger.error(f"Exception in {context}: {str(e)}")
    else:
        logger.error(f"Exception: {str(e)}")
    
    logger.debug(f"Traceback:\n{tb_text}")

def exception_handler(func):
    """Decorator to handle exceptions in functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except UpscalerError as e:
            # Already a custom error, just log it
            log_exception(e, func.__name__)
            raise
        except Exception as e:
            # Convert to custom error and log
            log_exception(e, func.__name__)
            raise UpscalerError(f"Unexpected error in {func.__name__}", e)
    return wrapper

def ui_exception_handler(func):
    """Decorator to handle exceptions in UI functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the exception
            log_exception(e, func.__name__)
            # Return False to indicate error (useful for UI callbacks)
            return False
    return wrapper
