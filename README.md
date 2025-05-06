# FreeAiImageUpscaler8k

<div align="center">
  <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#3498db;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#8e44ad;stop-opacity:1" />
      </linearGradient>
    </defs>
    <rect x="20" y="20" width="160" height="160" rx="15" fill="url(#grad1)"/>
    <rect x="40" y="40" width="120" height="120" rx="10" fill="#ffffff" fill-opacity="0.2"/>
    <path d="M60,100 L140,100 M100,60 L100,140" stroke="#ffffff" stroke-width="8" stroke-linecap="round"/>
    <circle cx="100" cy="100" r="30" fill="none" stroke="#ffffff" stroke-width="6"/>
    <text x="100" y="170" font-family="Arial" font-size="16" fill="#ffffff" text-anchor="middle">8K UPSCALER</text>
  </svg>
  <h3>Free AI-Powered Image Upscaling</h3>
  <p>Enhance your images to 8K quality with state-of-the-art AI models</p>

  <a href="https://buymeacoffee.com/rorrimaesu">
    <svg width="180" height="50" viewBox="0 0 180 50" xmlns="http://www.w3.org/2000/svg">
      <rect width="180" height="50" rx="10" fill="#FFDD00"/>
      <path d="M26,15 C26,15 28,13 30,15 C32,17 32,19 30,21 L26,25 L22,21 C20,19 20,17 22,15 C24,13 26,15 26,15 Z" fill="#FFFFFF"/>
      <rect x="23" y="24" width="6" height="10" rx="2" fill="#FFFFFF"/>
      <text x="90" y="30" font-family="Arial" font-size="14" font-weight="bold" fill="#000000" text-anchor="middle">Buy me a coffee</text>
    </svg>
  </a>
</div>

## 🌟 Features

- **High-Quality Upscaling**: Enhance images up to 4x their original resolution
- **Multiple AI Models**: Choose from different models optimized for photos or anime/illustrations
- **User-Friendly Interface**: Simple and intuitive GUI for easy operation
- **Batch Processing**: Upscale multiple images at once
- **Preview Functionality**: Compare original and upscaled images
- **Automatic Model Management**: Models are downloaded automatically when needed
- **GPU Acceleration**: Utilizes CUDA for faster processing (with CPU fallback)
- **Tiling Support**: Efficiently processes large images by breaking them into manageable tiles

## 📋 Requirements

- **Python 3.7+**
- **CUDA-compatible GPU** (recommended for faster processing)
- **Dependencies**: PyTorch, OpenCV, PyQt5, etc. (see `requirements.txt`)

## 🚀 Installation

### Option 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/RorriMaesu/FreeAiImageUpscaler8k.git
cd FreeAiImageUpscaler8k

# Create a virtual environment (optional but recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

### Option 2: Download the Release

1. Go to the [Releases](https://github.com/RorriMaesu/FreeAiImageUpscaler8k/releases) page
2. Download the latest release for your platform
3. Extract the archive and run the application

## 🖥️ Usage

### Running the Application

You can run the application in several ways:

```bash
# Option 1: Using the Python launcher script
python main.py

# Option 2: Using the batch file (Windows)
run_upscaler.bat

# Option 3: Running the main module directly
python imageUpscaler/main.py
```

### Using the GUI

1. **Select a Model**: Choose an AI model from the dropdown menu
   - **RealESRGAN x4plus**: General purpose 4x upscaler for photos
   - **RealESRGAN x4plus Anime**: Specialized 4x upscaler for anime/cartoon images
   - **RealESRGAN x2plus**: General purpose 2x upscaler for photos
   - **SwinIR Large**: High-quality 4x upscaler using Swin Transformer architecture

2. **Single Image Upscaling**:
   - Click "Select Image" to choose an image file
   - Click "Upscale Image" to process the image
   - The upscaled image will be displayed in the preview area and saved to the output directory

3. **Batch Processing**:
   - Click "Select Multiple Images" to choose multiple image files
   - Click "Batch Upscale" to process all selected images
   - Upscaled images will be saved to the output directory

## 🧠 AI Models

The application uses state-of-the-art AI models for image upscaling:

- **RealESRGAN**: Enhanced version of ESRGAN with improved training methodology and real-world degradation modeling
- **SwinIR**: Transformer-based model that achieves excellent results for image restoration tasks

Models are automatically downloaded when first used and stored in the `models` directory.

## 🔧 Technical Details

### Architecture

The application is built with a modular architecture:

- **Core Components**:
  - `models/model_manager.py`: Manages model loading and downloading
  - `models/upscaler.py`: Handles the upscaling process
  - `utils/image_processor.py`: Processes images with tiling for large images
  - `utils/config.py`: Manages application configuration
  - `utils/logger.py`: Centralized logging system
  - `utils/error_handler.py`: Custom error handling framework

- **UI Components**:
  - `ui/main_window.py`: Main application window
  - `ui/preview_widget.py`: Image preview widget with zoom capabilities

### Project Structure

```
FreeAiImageUpscaler8k/
├── assets/                  # Assets for the project (logos, etc.)
├── imageUpscaler/           # Main application package
│   ├── models/              # AI model implementations
│   │   ├── __init__.py
│   │   ├── model_manager.py # Model loading and management
│   │   ├── swinir_model.py  # SwinIR model implementation
│   │   └── upscaler.py      # Main upscaling logic
│   ├── ui/                  # User interface components
│   │   ├── __init__.py
│   │   ├── main_window.py   # Main application window
│   │   └── preview_widget.py # Image preview widget
│   ├── utils/               # Utility functions and classes
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration management
│   │   ├── downloader.py    # Model downloader
│   │   ├── error_handler.py # Error handling utilities
│   │   ├── image_processor.py # Image processing utilities
│   │   └── logger.py        # Logging utilities
│   ├── main.py              # Application entry point
│   ├── setup.py             # Setup script
│   └── test_*.py            # Test files
├── main.py                  # Root launcher script
├── run_upscaler.bat         # Windows batch launcher
├── LICENSE                  # MIT License
└── README.md                # This file
```

### Error Handling and Logging

The application implements a robust error handling and logging system:

- **Centralized Logging**: All components use a centralized logging system
- **Custom Exception Hierarchy**: Custom exception types for different error categories
- **Graceful Degradation**: The application attempts to continue operation when possible

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Support the Project

If you find this tool useful, consider supporting its development:

<div align="center">
  <a href="https://buymeacoffee.com/rorrimaesu">
    <svg width="180" height="50" viewBox="0 0 180 50" xmlns="http://www.w3.org/2000/svg">
      <rect width="180" height="50" rx="10" fill="#FFDD00"/>
      <path d="M26,15 C26,15 28,13 30,15 C32,17 32,19 30,21 L26,25 L22,21 C20,19 20,17 22,15 C24,13 26,15 26,15 Z" fill="#FFFFFF"/>
      <rect x="23" y="24" width="6" height="10" rx="2" fill="#FFFFFF"/>
      <text x="90" y="30" font-family="Arial" font-size="14" font-weight="bold" fill="#000000" text-anchor="middle">Buy me a coffee</text>
    </svg>
  </a>
</div>

Your support helps maintain and improve this free tool for everyone!

## 👥 Contributing

Contributions are welcome! Here's how you can contribute:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Commit your changes**:
   ```bash
   git commit -m 'Add some amazing feature'
   ```
5. **Push to the branch**:
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### Development Guidelines

- Follow the existing code style
- Add unit tests for new features
- Update documentation as needed
- Test your changes thoroughly

### Reporting Issues

If you encounter any problems or have suggestions for improvements:

1. Check if the issue already exists in the [Issues](https://github.com/RorriMaesu/FreeAiImageUpscaler8k/issues) section
2. If not, create a new issue with:
   - A clear title and description
   - Steps to reproduce the issue
   - Expected and actual behavior
   - Screenshots if applicable
   - System information (OS, Python version, GPU, etc.)

## 🔗 Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al.
- [SwinIR](https://github.com/JingyunLiang/SwinIR) by Jingyun Liang et al.
- [BasicSR](https://github.com/XPixelGroup/BasicSR) framework for Super-Resolution
