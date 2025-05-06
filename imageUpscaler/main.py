#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI Image Upscaler - Main Application
"""

import sys
import os
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox

from ui.main_window import MainWindow
from utils.config import Config
from utils.logger import Logger
from utils.error_handler import log_exception

def excepthook(exc_type, exc_value, exc_tb):
    """Global exception handler for unhandled exceptions"""
    # Format the exception
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    tb_text = ''.join(tb_lines)

    # Log the exception
    logger = Logger.get_logger()
    logger.critical(f"Unhandled exception: {exc_value}")
    logger.critical(f"Traceback:\n{tb_text}")

    # Show error message to user
    error_box = QMessageBox()
    error_box.setIcon(QMessageBox.Critical)
    error_box.setWindowTitle("Critical Error")
    error_box.setText("An unexpected error has occurred.")
    error_box.setInformativeText(str(exc_value))
    error_box.setDetailedText(tb_text)
    error_box.exec_()

def main():
    """Main application entry point"""
    try:
        # Initialize logger
        logger = Logger()
        log = logger.get_logger()
        log.info("Starting AI Image Upscaler application")

        # Set global exception handler
        sys.excepthook = excepthook

        # Create application configuration
        log.info("Loading configuration")
        config = Config()

        # Initialize the application
        log.info("Initializing Qt application")
        app = QApplication(sys.argv)
        app.setApplicationName("AI Image Upscaler")
        app.setOrganizationName("AI Image Upscaler")

        # Create and show the main window
        log.info("Creating main window")
        window = MainWindow(config)
        window.show()

        log.info("Application started successfully")

        # Start the application event loop
        return app.exec_()
    except Exception as e:
        # Log any exceptions during startup
        log_exception(e, "application startup")

        # Show error message
        if QApplication.instance():
            QMessageBox.critical(None, "Startup Error",
                                f"Error starting application: {str(e)}")

        return 1

if __name__ == "__main__":
    sys.exit(main())
