#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging utility for AI Image Upscaler
"""

import os
import logging
import datetime
from logging.handlers import RotatingFileHandler

class Logger:
    """Centralized logging configuration for the application"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one logger instance exists"""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, log_dir=None, log_level=logging.INFO):
        """Initialize logger with specified directory and level"""
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        self._initialized = True
        
        # Set default log directory if not provided
        if log_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(base_dir, 'logs')
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamp for log file name
        timestamp = datetime.datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(log_dir, f'upscaler_{timestamp}.log')
        
        # Configure root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)
        
        # Clear existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create file handler with rotation (10 MB max size, keep 5 backup files)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Logger initialized")
    
    @classmethod
    def get_logger(cls):
        """Get the configured logger instance"""
        if cls._instance is None:
            cls()
        return logging.getLogger()

def get_logger():
    """Convenience function to get the logger"""
    return Logger.get_logger()
