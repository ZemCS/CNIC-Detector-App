# CNIC Detection Flutter App

A Flutter mobile application for detecting and extracting information from Pakistani Computerized National Identity Cards (CNIC) using on-device machine learning and OCR.

## Overview

This project is forked from the [CNIC-Detection repository](https://github.com/your-username/CNIC-Detection) and has been adapted for mobile deployment. The app uses a nano version of the same YOLO model that has been converted to TensorFlow Lite (TFLite) format to enable all AI operations to work directly within the mobile application without requiring external API calls.

## Features

- **Real-time CNIC Detection**: Uses a custom-trained YOLO model to detect CNIC cards in various orientations
- **Dual-side Capture**: Captures both front and back sides of the CNIC with validation
- **Automatic Rotation**: Intelligently rotates detected CNIC regions based on orientation
- **OCR Text Extraction**: Extracts text from detected regions using Google ML Kit
- **Data Parsing**: Parses extracted text to identify specific CNIC fields
- **Validation**: Compares identity numbers between front and back to ensure authenticity
- **Offline Processing**: All operations run on-device without internet connectivity

## Supported CNIC Classes

The model can detect CNICs in the following orientations and states:

- `cnic` - Front side, normal orientation
- `cnic_back` - Back side, normal orientation  
- `cnic_back_number` - Back side number region
- `cnic_left` - Rotated 90° left
- `cnic_left_back` - Back side rotated 90° left
- `cnic_left_back_number` - Back side number region rotated 90° left
- `cnic_right` - Rotated 90° right
- `cnic_right_back` - Back side rotated 90° right
- `cnic_right_back_number` - Back side number region rotated 90° right
- `cnic_upside_down` - Rotated 180°
- `cnic_upside_down_back` - Back side rotated 180°
- `cnic_upside_down_back_number` - Back side number region rotated 180°

## Extracted Data Fields

The app extracts and displays the following CNIC information:

- **Identity Number** (13-digit format: XXXXX-XXXXXXX-X)
- **Name**
- **Father's Name**
- **Gender** (M/F)
- **Country of Stay**
- **Date of Birth**
- **Date of Issue**
- **Date of Expiry**

## Requirements

### Dependencies

```yaml
dependencies:
  flutter: ^3.0.0
  camera: ^0.10.0
  tflite_flutter: ^0.10.0
  image: ^4.0.0
  permission_handler: ^10.0.0
  google_ml_kit: ^0.13.0
  path_provider: ^2.0.0
```

### Model File

Place the TensorFlow Lite model file `best_float32.tflite` in the `assets/` directory of your Flutter project.

### Permissions

Add the following permissions to your platform-specific configuration files:

#### Android (`android/app/src/main/AndroidManifest.xml`)
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

#### iOS (`ios/Runner/Info.plist`)
```xml
<key>NSCameraUsageDescription</key>
<string>This app needs camera access to capture CNIC images</string>
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/cnic-detection-flutter.git
cd cnic-detection-flutter
```

2. Install dependencies:
```bash
flutter pub get
```

3. Add the TFLite model file to `assets/best_float32.tflite`

4. Update `pubspec.yaml` to include the model in assets:
```yaml
flutter:
  assets:
    - assets/best_float32.tflite
```

5. Run the app:
```bash
flutter run
```

## Usage

1. **Launch the App**: Open the application and wait for the camera and model to initialize
2. **Capture Front Side**: Position the CNIC front side within the camera view and tap the capture button
3. **Capture Back Side**: After successful front capture, flip the CNIC to the back side and capture again
4. **View Results**: The app will display the captured images with bounding boxes around detected regions
5. **Extract Data**: Tap "See Info" to view the parsed CNIC information
6. **Validation**: The app will validate the CNIC by comparing identity numbers from both sides

## Technical Details

### Model Architecture
- **Base Model**: YOLO (You Only Look Once) object detection
- **Format**: TensorFlow Lite (TFLite) for mobile optimization
- **Input Size**: 640x640 pixels
- **Output**: Bounding boxes with confidence scores and class predictions

### Image Processing Pipeline
1. **Capture**: High-resolution image capture using device camera
2. **Preprocessing**: Image resizing and normalization for model input
3. **Detection**: YOLO model inference to detect CNIC regions
4. **Post-processing**: Bounding box filtering and rotation correction
5. **OCR**: Google ML Kit text recognition on cropped regions
6. **Parsing**: Regular expression-based text parsing for structured data extraction

### Performance Optimizations
- **TFLite Model**: Optimized for mobile inference with reduced model size
- **On-device Processing**: No network dependency for core functionality
- **Memory Management**: Efficient image handling and cleanup
- **Camera Controls**: Optimized camera settings for document capture

## Code Structure

```
lib/
├── main.dart                 # App entry point and main widget
├── models/
│   ├── parsed_cnic_data.dart # Data model for CNIC information
│   └── detection.dart        # Detection result model
├── services/
│   └── cnic_detection_service.dart # Core ML and OCR service
├── screens/
│   ├── camera_screen.dart    # Camera capture interface
│   ├── results_screen.dart   # Results display with bounding boxes
│   └── data_screen.dart      # Extracted data display
└── widgets/
    └── bounding_box_painter.dart # Custom painter for detection visualization
```

## Known Limitations

- **Lighting Conditions**: Performance may vary under poor lighting
- **CNIC Condition**: Damaged or heavily worn CNICs may affect detection accuracy
- **Text Quality**: OCR accuracy depends on image clarity and text legibility
- **Language Support**: Optimized for English text recognition

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original YOLO model training and dataset preparation
- Google ML Kit for OCR capabilities
- Flutter and TensorFlow Lite teams for mobile ML framework
- Contributors to the original CNIC-Detection repository

## Support

For issues and questions:
1. Check the [Issues](https://github.com/your-username/cnic-detection-flutter/issues) page
2. Create a new issue with detailed description and screenshots
3. Provide device information and Flutter version for better support
