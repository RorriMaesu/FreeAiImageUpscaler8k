#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main window for AI Image Upscaler
"""

import os
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog, QProgressBar, QMessageBox, QCheckBox, QGroupBox,
    QScrollArea, QSizePolicy, QSpacerItem, QApplication
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

from models.upscaler import Upscaler
from ui.preview_widget import ImagePreviewWidget

class UpscalerThread(QThread):
    """Thread for running upscaling operations"""
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, upscaler, image_path, output_path, model_name):
        super().__init__()
        self.upscaler = upscaler
        self.image_path = image_path
        self.output_path = output_path
        self.model_name = model_name
    
    def run(self):
        try:
            output_path = self.upscaler.upscale_image(
                self.image_path,
                self.output_path,
                self.model_name,
                lambda progress: self.progress_signal.emit(progress)
            )
            self.finished_signal.emit(output_path)
        except Exception as e:
            self.error_signal.emit(str(e))

class BatchUpscalerThread(QThread):
    """Thread for running batch upscaling operations"""
    progress_signal = pyqtSignal(int, int, int)  # progress, current, total
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    
    def __init__(self, upscaler, image_paths, output_dir, model_name):
        super().__init__()
        self.upscaler = upscaler
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.model_name = model_name
    
    def run(self):
        try:
            output_paths = self.upscaler.upscale_batch(
                self.image_paths,
                self.output_dir,
                self.model_name,
                lambda progress, current, total: self.progress_signal.emit(progress, current, total)
            )
            self.finished_signal.emit(output_paths)
        except Exception as e:
            self.error_signal.emit(str(e))

class DownloadModelThread(QThread):
    """Thread for downloading models"""
    progress_signal = pyqtSignal(str, int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, upscaler, model_name):
        super().__init__()
        self.upscaler = upscaler
        self.model_name = model_name
    
    def run(self):
        try:
            model_path = self.upscaler.download_model(
                self.model_name,
                lambda model_name, progress: self.progress_signal.emit(model_name, progress)
            )
            self.finished_signal.emit(model_path)
        except Exception as e:
            self.error_signal.emit(str(e))

class MainWindow(QMainWindow):
    """Main window for the application"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.upscaler = Upscaler(config)
        self.current_image_path = None
        self.current_output_path = None
        self.batch_image_paths = []
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Set window properties
        self.setWindowTitle("AI Image Upscaler")
        self.setMinimumSize(800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout(model_group)
        
        self.model_combo = QComboBox()
        self.refresh_model_list()
        
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_combo, 1)
        
        self.download_button = QPushButton("Download Model")
        self.download_button.clicked.connect(self.download_model)
        model_layout.addWidget(self.download_button)
        
        main_layout.addWidget(model_group)
        
        # Create image preview area
        preview_group = QGroupBox("Image Preview")
        preview_layout = QHBoxLayout(preview_group)
        
        # Input image preview
        input_preview_layout = QVBoxLayout()
        input_preview_layout.addWidget(QLabel("Input Image:"))
        self.input_preview = ImagePreviewWidget()
        input_preview_layout.addWidget(self.input_preview)
        
        # Output image preview
        output_preview_layout = QVBoxLayout()
        output_preview_layout.addWidget(QLabel("Output Image:"))
        self.output_preview = ImagePreviewWidget()
        output_preview_layout.addWidget(self.output_preview)
        
        preview_layout.addLayout(input_preview_layout)
        preview_layout.addLayout(output_preview_layout)
        
        main_layout.addWidget(preview_group)
        
        # Create controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Single image controls
        single_image_layout = QHBoxLayout()
        
        self.select_image_button = QPushButton("Select Image")
        self.select_image_button.clicked.connect(self.select_image)
        single_image_layout.addWidget(self.select_image_button)
        
        self.upscale_button = QPushButton("Upscale Image")
        self.upscale_button.clicked.connect(self.upscale_image)
        self.upscale_button.setEnabled(False)
        single_image_layout.addWidget(self.upscale_button)
        
        controls_layout.addLayout(single_image_layout)
        
        # Batch processing controls
        batch_layout = QHBoxLayout()
        
        self.select_batch_button = QPushButton("Select Multiple Images")
        self.select_batch_button.clicked.connect(self.select_batch_images)
        batch_layout.addWidget(self.select_batch_button)
        
        self.batch_upscale_button = QPushButton("Batch Upscale")
        self.batch_upscale_button.clicked.connect(self.batch_upscale)
        self.batch_upscale_button.setEnabled(False)
        batch_layout.addWidget(self.batch_upscale_button)
        
        controls_layout.addLayout(batch_layout)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        controls_layout.addLayout(progress_layout)
        
        # Status label
        self.status_label = QLabel("Ready")
        controls_layout.addWidget(self.status_label)
        
        main_layout.addWidget(controls_group)
        
        # Set central widget
        self.setCentralWidget(central_widget)
    
    def refresh_model_list(self):
        """Refresh the list of available models"""
        self.model_combo.clear()
        
        available_models = self.upscaler.get_available_models()
        for model_id, model_info in available_models.items():
            self.model_combo.addItem(model_info.get('name', model_id), model_id)
    
    def select_image(self):
        """Select an image file for upscaling"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.input_preview.set_image(file_path)
            self.upscale_button.setEnabled(True)
            self.status_label.setText(f"Selected image: {os.path.basename(file_path)}")
    
    def select_batch_images(self):
        """Select multiple image files for batch upscaling"""
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(
            self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_paths:
            self.batch_image_paths = file_paths
            self.batch_upscale_button.setEnabled(True)
            self.status_label.setText(f"Selected {len(file_paths)} images for batch processing")
    
    def upscale_image(self):
        """Upscale the selected image"""
        if not self.current_image_path:
            return
        
        # Get selected model
        model_id = self.model_combo.currentData()
        
        # Generate output path
        filename, ext = os.path.splitext(os.path.basename(self.current_image_path))
        output_dir = self.config.get_output_dir()
        output_path = os.path.join(output_dir, f"{filename}_upscaled{ext}")
        
        # Disable UI during processing
        self.upscale_button.setEnabled(False)
        self.select_image_button.setEnabled(False)
        self.batch_upscale_button.setEnabled(False)
        self.select_batch_button.setEnabled(False)
        self.download_button.setEnabled(False)
        
        # Start upscaling thread
        self.upscaler_thread = UpscalerThread(
            self.upscaler, self.current_image_path, output_path, model_id
        )
        self.upscaler_thread.progress_signal.connect(self.update_progress)
        self.upscaler_thread.finished_signal.connect(self.upscale_finished)
        self.upscaler_thread.error_signal.connect(self.upscale_error)
        self.upscaler_thread.start()
        
        self.status_label.setText("Upscaling image...")
    
    def batch_upscale(self):
        """Upscale multiple images in batch mode"""
        if not self.batch_image_paths:
            return
        
        # Get selected model
        model_id = self.model_combo.currentData()
        
        # Get output directory
        output_dir = self.config.get_output_dir()
        
        # Disable UI during processing
        self.upscale_button.setEnabled(False)
        self.select_image_button.setEnabled(False)
        self.batch_upscale_button.setEnabled(False)
        self.select_batch_button.setEnabled(False)
        self.download_button.setEnabled(False)
        
        # Start batch upscaling thread
        self.batch_thread = BatchUpscalerThread(
            self.upscaler, self.batch_image_paths, output_dir, model_id
        )
        self.batch_thread.progress_signal.connect(self.update_batch_progress)
        self.batch_thread.finished_signal.connect(self.batch_upscale_finished)
        self.batch_thread.error_signal.connect(self.upscale_error)
        self.batch_thread.start()
        
        self.status_label.setText("Batch upscaling images...")
    
    def download_model(self):
        """Download the selected model"""
        # Get selected model
        model_id = self.model_combo.currentData()
        
        # Disable UI during download
        self.upscale_button.setEnabled(False)
        self.select_image_button.setEnabled(False)
        self.batch_upscale_button.setEnabled(False)
        self.select_batch_button.setEnabled(False)
        self.download_button.setEnabled(False)
        
        # Start download thread
        self.download_thread = DownloadModelThread(self.upscaler, model_id)
        self.download_thread.progress_signal.connect(self.update_download_progress)
        self.download_thread.finished_signal.connect(self.download_finished)
        self.download_thread.error_signal.connect(self.download_error)
        self.download_thread.start()
        
        self.status_label.setText(f"Downloading model: {model_id}...")
    
    def update_progress(self, progress):
        """Update progress bar for single image upscaling"""
        self.progress_bar.setValue(progress)
    
    def update_batch_progress(self, progress, current, total):
        """Update progress bar for batch upscaling"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Upscaling image {current}/{total}...")
    
    def update_download_progress(self, model_name, progress):
        """Update progress bar for model download"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Downloading model: {model_name} ({progress}%)...")
    
    def upscale_finished(self, output_path):
        """Handle completion of upscaling process"""
        # Re-enable UI
        self.upscale_button.setEnabled(True)
        self.select_image_button.setEnabled(True)
        self.batch_upscale_button.setEnabled(bool(self.batch_image_paths))
        self.select_batch_button.setEnabled(True)
        self.download_button.setEnabled(True)
        
        # Update preview
        self.current_output_path = output_path
        self.output_preview.set_image(output_path)
        
        # Update status
        self.status_label.setText(f"Upscaling complete: {os.path.basename(output_path)}")
        
        # Show success message
        QMessageBox.information(self, "Upscaling Complete", 
                               f"Image has been upscaled and saved to:\n{output_path}")
    
    def batch_upscale_finished(self, output_paths):
        """Handle completion of batch upscaling process"""
        # Re-enable UI
        self.upscale_button.setEnabled(bool(self.current_image_path))
        self.select_image_button.setEnabled(True)
        self.batch_upscale_button.setEnabled(True)
        self.select_batch_button.setEnabled(True)
        self.download_button.setEnabled(True)
        
        # Update status
        self.status_label.setText(f"Batch upscaling complete: {len(output_paths)} images processed")
        
        # Show success message
        QMessageBox.information(self, "Batch Upscaling Complete", 
                               f"{len(output_paths)} images have been upscaled and saved to:\n{self.config.get_output_dir()}")
    
    def download_finished(self, model_path):
        """Handle completion of model download"""
        # Re-enable UI
        self.upscale_button.setEnabled(bool(self.current_image_path))
        self.select_image_button.setEnabled(True)
        self.batch_upscale_button.setEnabled(bool(self.batch_image_paths))
        self.select_batch_button.setEnabled(True)
        self.download_button.setEnabled(True)
        
        # Update status
        self.status_label.setText(f"Model download complete: {os.path.basename(model_path)}")
        
        # Show success message
        QMessageBox.information(self, "Download Complete", 
                               f"Model has been downloaded and saved to:\n{model_path}")
    
    def upscale_error(self, error_message):
        """Handle errors during upscaling"""
        # Re-enable UI
        self.upscale_button.setEnabled(bool(self.current_image_path))
        self.select_image_button.setEnabled(True)
        self.batch_upscale_button.setEnabled(bool(self.batch_image_paths))
        self.select_batch_button.setEnabled(True)
        self.download_button.setEnabled(True)
        
        # Update status
        self.status_label.setText(f"Error: {error_message}")
        
        # Show error message
        QMessageBox.critical(self, "Upscaling Error", 
                            f"An error occurred during upscaling:\n{error_message}")
    
    def download_error(self, error_message):
        """Handle errors during model download"""
        # Re-enable UI
        self.upscale_button.setEnabled(bool(self.current_image_path))
        self.select_image_button.setEnabled(True)
        self.batch_upscale_button.setEnabled(bool(self.batch_image_paths))
        self.select_batch_button.setEnabled(True)
        self.download_button.setEnabled(True)
        
        # Update status
        self.status_label.setText(f"Download error: {error_message}")
        
        # Show error message
        QMessageBox.critical(self, "Download Error", 
                            f"An error occurred during model download:\n{error_message}")
