#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image preview widget for AI Image Upscaler
"""

import os
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QScrollArea, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize

class ImagePreviewWidget(QScrollArea):
    """Widget for displaying image previews with zoom and scroll capabilities"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up scroll area
        self.setWidgetResizable(True)
        self.setMinimumSize(300, 300)
        
        # Create image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setScaledContents(True)
        
        # Set image label as widget for scroll area
        self.setWidget(self.image_label)
        
        # Initialize variables
        self.current_image_path = None
        self.pixmap = None
        self.zoom_factor = 1.0
        
        # Set up empty preview
        self.clear_image()
    
    def set_image(self, image_path):
        """Set the image to display"""
        if not image_path or not os.path.exists(image_path):
            self.clear_image()
            return
        
        # Load image
        self.pixmap = QPixmap(image_path)
        if self.pixmap.isNull():
            self.clear_image()
            return
        
        # Store image path
        self.current_image_path = image_path
        
        # Reset zoom
        self.zoom_factor = 1.0
        
        # Update display
        self.update_display()
    
    def clear_image(self):
        """Clear the displayed image"""
        self.current_image_path = None
        self.pixmap = None
        self.image_label.setText("No image")
        self.image_label.setStyleSheet("QLabel { background-color : #f0f0f0; }")
    
    def update_display(self):
        """Update the displayed image with current zoom factor"""
        if self.pixmap is None:
            return
        
        # Calculate scaled size
        scaled_size = self.pixmap.size() * self.zoom_factor
        
        # Set minimum size of label to scaled size
        self.image_label.setMinimumSize(scaled_size)
        
        # Set pixmap
        self.image_label.setPixmap(self.pixmap.scaled(
            scaled_size, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))
    
    def zoom_in(self):
        """Zoom in on the image"""
        if self.pixmap is None:
            return
        
        self.zoom_factor *= 1.25
        self.update_display()
    
    def zoom_out(self):
        """Zoom out from the image"""
        if self.pixmap is None:
            return
        
        self.zoom_factor *= 0.8
        self.update_display()
    
    def reset_zoom(self):
        """Reset zoom to original size"""
        if self.pixmap is None:
            return
        
        self.zoom_factor = 1.0
        self.update_display()
    
    def fit_to_view(self):
        """Fit the image to the view"""
        if self.pixmap is None:
            return
        
        # Calculate zoom factor to fit view
        view_size = self.viewport().size()
        image_size = self.pixmap.size()
        
        width_ratio = view_size.width() / image_size.width()
        height_ratio = view_size.height() / image_size.height()
        
        # Use the smaller ratio to ensure the entire image is visible
        self.zoom_factor = min(width_ratio, height_ratio)
        
        self.update_display()
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming"""
        if self.pixmap is None:
            return
        
        # Get delta
        delta = event.angleDelta().y()
        
        # Zoom in or out based on wheel direction
        if delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor *= 0.9
        
        self.update_display()
    
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        
        # Fit image to view when widget is resized
        self.fit_to_view()
