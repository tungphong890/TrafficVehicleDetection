# Traffic Vehicle Detection

Advanced real-time vehicle detection system using state-of-the-art deep learning models (YOLOv8, Faster R-CNN, SSD-VGG16) to identify vehicles, number plates, and traffic participants in images and videos.

## Overview

This project implements a comprehensive traffic and vehicle detection pipeline designed for intelligent transportation systems, parking management, toll systems, and traffic analysis. It leverages cutting-edge computer vision models to accurately detect and classify various types of vehicles and objects on the road.

### Key Features
-  **Real-time Detection**: Process images and videos at high speeds
-  **Multiple Model Support**: YOLOv8, Faster R-CNN, and SSD-VGG16
-  **8 Object Classes**: Cars, Number Plates, Blurred Plates, Two-Wheelers, Autos, Buses, and Trucks
-  **Web Interface**: User-friendly Flask web application
-  **Video Processing**: Handle both images and video files with preview generation
-  **Confidence Scores**: Display prediction confidence for each detection
-  **File Management**: Automatic cleanup and efficient storage

##  Architecture & Project Structure

```
TrafficVehicleDetection/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── Faster_RCNN                     # Faster R-CNN model notebook
├── YOLOv8                          # Pre-trained YOLOv8 model file
├── sdd-vgg16.ipynb                 # SSD with VGG16 implementation
├── templates/                      # HTML templates for web UI
│   ├── index.html                 # Main upload interface
│   ├── result.html                # Results display page
│   └── gallery.html               # Previous results gallery
└── static/
    ├── uploads/                    # User uploaded files
    ├── results/                    # Processed output images/videos
    └── css/                        # Styling assets
```

### Model Components

1. **YOLOv8** (Primary Model)
   - Real-time object detection architecture
   - Optimized for fast inference on CPU/GPU
   - Trained on custom traffic dataset
   - Detects 8 vehicle classes

2. **Faster R-CNN**
   - Two-stage detector for high accuracy
   - Fine-tuned for vehicle detection
   - Slower but more accurate than YOLO

3. **SSD-VGG16**
   - Single Shot MultiBox Detector
   - Good balance between speed and accuracy
   - Efficient for edge devices

##  Installation Guide

### Prerequisites
- **Python**: 3.8 or higher
- **System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB for models

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/tungphong890/TrafficVehicleDetection.git
cd TrafficVehicleDetection
```

#### 2. Create Virtual Environment
```bash
# Using venv (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Detailed dependency information:**
- **Flask** (≥2.0): Web framework for the interface
- **PyTorch & TorchVision**: Deep learning framework and computer vision utilities
- **Ultralytics**: YOLOv8 implementation and utilities
- **OpenCV**: Image and video processing
- **Pillow**: Image manipulation

#### 4. Download Pre-trained Models
```bash
# YOLOv8 model (automatically downloaded on first run)
# Or manually download from:
# https://drive.google.com/drive/folders/1Jy8mxYi99WZO3vv5iPqKbxyP5aX8_L84?hl=vi

# Place the model file in the project root:
cp yolov8.pt ./
```

#### 5. Verify Installation
```bash
python -c "from ultralytics import YOLO; print('YOLOv8 ready!')"
```

### Installation Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: ultralytics` | Run `pip install --upgrade ultralytics` |
| CUDA/GPU issues | Ensure PyTorch CUDA version matches your GPU; use CPU fallback in app.py |
| Model file not found | Download from Google Drive link and place in project root |
| Port 5000 already in use | Change `app.run(port=5001)` in app.py |

##  Usage Guide

### Running the Web Application

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

### Web Interface Features

1. **Upload Page**
   - Drag-and-drop file upload
   - Support for PNG, JPG, JPEG, MP4
   - File size validation
   - Real-time upload feedback

2. **Detection Page**
   - View detection results with bounding boxes
   - Display confidence scores
   - Object class labels
   - Result download options

3. **Gallery Page**
   - View previous detection results
   - Download results
   - Delete old files

### Command-Line Usage (Advanced)

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8.pt')

# Detect objects in image
results = model.predict(source='path/to/image.jpg', conf=0.25)

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])
        print(f"Detected: {cls} with confidence {conf:.2f}")
```

##  Model Performance & Class Mapping

### Detected Classes
```
Index | Class Name         | Use Case
------|-------------------|----------------------------------
  0   | Background        | (Ignored detections)
  1   | Car               | Sedans, SUVs, hatchbacks
  2   | Number Plate      | Visible registration plates
  3   | Blur Number Plate | Obscured/motion-blurred plates
  4   | Two Wheeler       | Motorcycles, scooters
  5   | Auto              | Auto-rickshaws
  6   | Bus               | Public transport buses
  7   | Truck             | Cargo trucks, commercial vehicles
```

### Detection Parameters
- **Confidence Threshold**: 0.25 (adjustable)
- **Input Size**: 640x640 pixels (YOLOv8)
- **Device**: CPU (GPU support available)
- **Processing Speed**: ~10-50 FPS depending on resolution

##  Best Practices

### For Optimal Results

1. **Image Quality**
   - Use clear, well-lit images
   - Minimum 480p resolution recommended
   - Avoid extreme angles or blur
   - Daylight conditions work best

2. **Video Processing**
   - Videos should be <500MB
   - 30 FPS or higher recommended
   - H.264 codec preferred
   - Duration: <5 minutes for quick processing

3. **Confidence Tuning**
   ```python
   # In app.py, modify the confidence threshold:
   results = model.predict(source=image_path, conf=0.35)  # Higher = stricter
   ```

4. **Performance Optimization**
   - Use smaller input sizes for faster processing
   - Batch process multiple images
   - Cache model loading
   - Use GPU if available

5. **Data Management**
   - Regular cleanup of upload/result folders
   - Archive old results periodically
   - Monitor disk space usage

##  Configuration Guide

### Modify Detection Settings

Edit `app.py` to customize:

```python
# Line 16: Change allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}

# Line 19: Update class names
CLASS_NAMES = ['__background__', 'Car', 'Number Plate', ...]

# Line 53: Adjust confidence threshold
conf=0.25  # Change this value (0-1)

# Line 30: Modify image size
imgsz=640  # Can be 320, 416, 512, 640, etc.
```

### Web Server Configuration

```python
# In app.py
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',      # Listen on all interfaces
        port=5000,           # Change port number
        debug=False,         # Set to True for development
        threaded=True        # Enable threading
    )
```

##  Training Your Own Model

To train on custom data:

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8m.pt')

# Train on custom dataset
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    device=0  # GPU index or 0 for CPU
)
```

##  Troubleshooting

### Common Issues

**Issue**: Detection accuracy is low
- **Solution**: Ensure good lighting, check image quality, adjust confidence threshold

**Issue**: Application crashes on large videos
- **Solution**: Process smaller video segments, reduce resolution, increase available RAM

**Issue**: GPU not being utilized
- **Solution**: Install GPU-enabled PyTorch, verify CUDA installation

**Issue**: Slow processing speed
- **Solution**: Reduce image size (imgsz parameter), use smaller model (YOLOv8n), enable GPU

##  Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Docs](https://pytorch.org/docs/)
- [OpenCV Tutorials](https://docs.opencv.org/)
- [Trained Models Drive](https://drive.google.com/drive/folders/1Jy8mxYi99WZO3vv5iPqKbxyP5aX8_L84?hl=vi)

##  License

This project is open source and available under the MIT License.

##  Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Share feedback

##  Acknowledgments

- Ultralytics for YOLOv8
- PyTorch community for deep learning framework
- Contributors and users providing feedback

---

**Last Updated**: March 2026  
**Version**: 1.0.0
