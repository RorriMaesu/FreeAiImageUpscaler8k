#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model downloader utility for AI Image Upscaler
"""

import os
import requests
import time
from tqdm import tqdm

from utils.error_handler import NetworkError, exception_handler
from utils.logger import get_logger

logger = get_logger()

class ModelDownloader:
    """Utility for downloading pre-trained models"""

    @exception_handler
    def __init__(self, models_dir):
        """Initialize downloader with models directory"""
        self.models_dir = models_dir
        logger.debug(f"Initializing ModelDownloader with directory: {models_dir}")
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            logger.debug(f"Ensured models directory exists: {models_dir}")
        except Exception as e:
            error_msg = f"Failed to create models directory: {e}"
            logger.error(error_msg)
            raise NetworkError(error_msg, e)

    @exception_handler
    def download_model(self, model_name, model_url, callback=None):
        """
        Download a model from the given URL

        Args:
            model_name: Name of the model
            model_url: URL to download the model from
            callback: Optional callback function to report progress

        Returns:
            Path to the downloaded model file
        """
        logger.info(f"Downloading model: {model_name} from {model_url}")

        # Create model file path
        model_path = os.path.join(self.models_dir, f"{model_name}.pth")

        # Check if model already exists
        if os.path.exists(model_path):
            logger.info(f"Model already exists: {model_path}")
            if callback:
                callback(model_name, 100)
            return model_path

        # Download the model
        try:
            # Create session with retry mechanism
            session = requests.Session()
            retries = 3
            retry_delay = 2  # seconds

            for attempt in range(retries):
                try:
                    logger.debug(f"Download attempt {attempt+1}/{retries}")
                    response = session.get(model_url, stream=True, timeout=(10, 30))
                    response.raise_for_status()
                    break
                except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                    if attempt < retries - 1:
                        logger.warning(f"Download attempt {attempt+1} failed: {e}. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise

            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            logger.debug(f"Model file size: {total_size/1024/1024:.2f} MB")
            block_size = 1024 * 8  # 8 KB

            # Create progress bar
            progress_bar = tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {model_name}"
            )

            # Download and save the file
            with open(model_path, 'wb') as f:
                downloaded = 0
                for data in response.iter_content(block_size):
                    f.write(data)
                    progress_bar.update(len(data))
                    downloaded += len(data)

                    # Report progress if callback is provided
                    if callback and total_size > 0:
                        progress = int(downloaded / total_size * 100)
                        callback(model_name, progress)

            progress_bar.close()

            # Final progress update
            if callback:
                callback(model_name, 100)

            logger.info(f"Model downloaded successfully: {model_path}")
            return model_path

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error downloading model {model_name}: {e}"
            logger.error(error_msg)
            # Remove partially downloaded file if it exists
            if os.path.exists(model_path):
                logger.debug(f"Removing partial download: {model_path}")
                os.remove(model_path)
            raise NetworkError(error_msg, e)
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error downloading model {model_name}: {e}"
            logger.error(error_msg)
            if os.path.exists(model_path):
                logger.debug(f"Removing partial download: {model_path}")
                os.remove(model_path)
            raise NetworkError(error_msg, e)
        except requests.exceptions.Timeout as e:
            error_msg = f"Timeout downloading model {model_name}: {e}"
            logger.error(error_msg)
            if os.path.exists(model_path):
                logger.debug(f"Removing partial download: {model_path}")
                os.remove(model_path)
            raise NetworkError(error_msg, e)
        except Exception as e:
            error_msg = f"Error downloading model {model_name}: {e}"
            logger.error(error_msg)
            # Remove partially downloaded file if it exists
            if os.path.exists(model_path):
                logger.debug(f"Removing partial download: {model_path}")
                os.remove(model_path)
            raise NetworkError(error_msg, e)

    @exception_handler
    def check_model_exists(self, model_name):
        """Check if a model file exists"""
        model_path = os.path.join(self.models_dir, f"{model_name}.pth")
        exists = os.path.exists(model_path)
        logger.debug(f"Checking if model exists: {model_name} - {'Found' if exists else 'Not found'}")
        return exists

    @exception_handler
    def get_model_path(self, model_name):
        """Get the path to a model file"""
        model_path = os.path.join(self.models_dir, f"{model_name}.pth")
        logger.debug(f"Getting model path for {model_name}: {model_path}")
        return model_path
