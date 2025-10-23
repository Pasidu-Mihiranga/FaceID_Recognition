# Face ID System

A comprehensive face recognition system with continuous learning capabilities, built using state-of-the-art computer vision and machine learning technologies.

## 🌟 Features

### Core Capabilities
- **Face Registration**: Upload and register faces with names
- **Face Recognition**: Real-time face identification from images or camera
- **Continuous Learning**: Automatically improves recognition with new encounters
- **Multiple Models**: Support for various detection and recognition algorithms
- **Database Management**: SQLite-based storage with comprehensive face data
- **Web Interface**: User-friendly web interface for easy management

### Supported Technologies

#### Face Detection Models
- **MTCNN**: Multi-task CNN for accurate face detection and alignment
- **RetinaFace**: Single-shot multi-level face localization
- **OpenCV Haar Cascade**: Fast and reliable face detection
- **Dlib**: HOG-based face detector

#### Face Recognition Models
- **ArcFace**: Additive Angular Margin Loss for deep face recognition
- **FaceNet**: Google's face recognition system
- **VGG-Face**: Oxford VGG's face recognition model

#### Additional Features
- **Real-time Camera Integration**: Live face recognition from webcam
- **Web Interface**: Flask-based web application
- **Continuous Learning**: Automatic model updates from new data
- **Database Analytics**: Comprehensive statistics and reporting
- **Export/Import**: Data backup and migration capabilities

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FaceID/FID
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the system**
   ```bash
   # Start with web interface
   python face_id_system.py --web
   
   # Start camera recognition
   python face_id_system.py --camera
   
   # Interactive mode
   python face_id_system.py
   ```

### Web Interface

Access the web interface at `http://localhost:5000` to:
- Register new persons
- Recognize faces from uploaded images
- View system statistics and analytics
- Manage registered persons
- Monitor continuous learning progress

## 📖 Usage Examples

### Basic Usage

```python
from main import FaceIDSystem

# Initialize the system
face_id = FaceIDSystem(
    detector_type='opencv',      # Face detector
    recognition_model='arcface', # Recognition model
    recognition_threshold=0.6    # Recognition threshold
)

# Register a person
face_id.register_person('path/to/image.jpg', 'John Doe')

# Recognize a face
import cv2
image = cv2.imread('path/to/test_image.jpg')
person_name, confidence, face_info = face_id.recognize_face(image)

print(f"Recognized: {person_name} (confidence: {confidence:.3f})")
```

### Camera Recognition

```python
# Start real-time camera recognition
face_id.start_camera_recognition()

# The system will automatically:
# - Detect faces in the camera feed
# - Recognize known persons
# - Learn from new encounters
# - Display results in real-time
```

### Continuous Learning

```python
# The system automatically learns from high-confidence recognitions
# You can also manually trigger learning updates:

# Process a recognition result for learning
face_id.learning_manager.process_recognition_result(
    face_image, person_name, confidence
)

# Get learning statistics
stats = face_id.learning_manager.get_learning_stats()
print(f"Learning updates: {stats['total_updates']}")
```

## 🏗️ System Architecture

```
Face ID System/
├── src/
│   ├── face_detection/     # Face detection modules
│   ├── face_recognition/   # Face recognition modules
│   ├── database/          # Database management
│   ├── continuous_learning/ # Learning algorithms
│   └── web_interface/     # Web application
├── data/
│   ├── registered_faces/   # Stored face images
│   ├── embeddings/        # Face embeddings
│   └── face_database.db   # SQLite database
├── models/                # Pre-trained models
├── main.py               # Main system integration
├── face_id_system.py     # Command-line interface
└── examples.py           # Usage examples
```

## 🔧 Configuration

### Command Line Options

```bash
python face_id_system.py [OPTIONS]

Options:
  --detector {mtcnn,retinaface,opencv,dlib}
                        Face detector type (default: opencv)
  --model {arcface,facenet,vggface}
                        Face recognition model (default: arcface)
  --threshold FLOAT     Recognition threshold (default: 0.6)
  --web                 Start web interface
  --camera              Start camera recognition
  --register TEXT       Register person: 'image_path,name'
  --port INTEGER        Web interface port (default: 5000)
  --host TEXT           Web interface host (default: 0.0.0.0)
```

### System Parameters

- **Recognition Threshold**: Controls how strict the recognition is (0.0-1.0)
- **Learning Threshold**: Minimum confidence for continuous learning
- **Max Embeddings**: Maximum embeddings stored per person
- **Camera Settings**: Resolution, FPS, and detection intervals

## 📊 Performance Metrics

### Model Performance

| Model | Accuracy | Speed | Memory Usage |
|-------|----------|-------|--------------|
| ArcFace | 96.7% | Medium | High |
| FaceNet | 97.4% | Fast | Medium |
| VGG-Face | 96.7% | Slow | High |

### Detection Performance

| Detector | Accuracy | Speed | Robustness |
|----------|----------|-------|------------|
| MTCNN | High | Slow | High |
| RetinaFace | Very High | Medium | Very High |
| OpenCV | Medium | Very Fast | Medium |
| Dlib | High | Fast | Medium |

## 🔒 Privacy and Security

- **Local Processing**: All face recognition happens locally
- **Data Encryption**: Face embeddings are stored securely
- **No Cloud Dependencies**: Complete offline operation
- **User Control**: Full control over data storage and deletion

## 🛠️ Development

### Running Tests

```bash
# Run example scripts
python examples.py

# Test individual components
python -m src.face_detection
python -m src.face_recognition
python -m src.database
```

### Adding New Models

1. Create a new recognizer class in `src/face_recognition/`
2. Implement the `FaceRecognizer` interface
3. Add the model to the factory function
4. Update the command-line options

### Customizing the Web Interface

- Templates: `src/web_interface/templates/`
- Static files: `src/web_interface/static/`
- Routes: `src/web_interface/__init__.py`

## 📚 API Reference

### FaceIDSystem Class

```python
class FaceIDSystem:
    def __init__(self, detector_type='mtcnn', recognition_model='arcface', 
                 recognition_threshold=0.6, learning_threshold=0.7)
    
    def register_person(self, image_path, person_name, metadata=None)
    def recognize_face(self, image)
    def start_camera_recognition(self, camera_index=0, display_window=True)
    def stop_camera_recognition(self)
    def get_system_stats(self)
    def export_system_data(self, export_path)
    def cleanup_system(self)
```

### Web API Endpoints

- `POST /api/register` - Register a new person
- `POST /api/recognize` - Recognize a face from uploaded image
- `POST /api/recognize_base64` - Recognize from base64 image data
- `GET /api/stats` - Get system statistics
- `GET /api/persons` - Get all registered persons
- `DELETE /api/person/<id>` - Delete a person
- `POST /api/camera/start` - Start camera recognition
- `POST /api/camera/stop` - Stop camera recognition

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FaceNet**: Google's face recognition system
- **ArcFace**: DeepInsight's face recognition framework
- **MTCNN**: Multi-task CNN for face detection
- **RetinaFace**: Single-shot face detection
- **DeepFace**: Lightweight face recognition framework

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the example scripts

## 🔮 Future Enhancements

- [ ] Mobile app integration
- [ ] Cloud deployment options
- [ ] Advanced analytics dashboard
- [ ] Multi-camera support
- [ ] Face anti-spoofing
- [ ] Age and gender estimation
- [ ] Emotion recognition
- [ ] Batch processing capabilities
