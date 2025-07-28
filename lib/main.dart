import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'package:google_ml_kit/google_ml_kit.dart';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Get available cameras
  final cameras = await availableCameras();
  
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  
  const MyApp({Key? key, required this.cameras}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CNIC Detection',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: CameraScreen(cameras: cameras),
    );
  }
}

// Class to hold parsed CNIC data
class ParsedCnicData {
  final String name;
  final String fatherName;
  final String gender;
  final String countryOfStay;
  final String identityNumber;
  final String dateOfBirth;
  final String dateOfIssue;
  final String dateOfExpiry;

  ParsedCnicData({
    this.name = '',
    this.fatherName = '',
    this.gender = '',
    this.countryOfStay = '',
    this.identityNumber = '',
    this.dateOfBirth = '',
    this.dateOfIssue = '',
    this.dateOfExpiry = '',
  });

  bool get hasAnyData => 
    name.isNotEmpty || 
    fatherName.isNotEmpty || 
    gender.isNotEmpty || 
    countryOfStay.isNotEmpty || 
    identityNumber.isNotEmpty || 
    dateOfBirth.isNotEmpty || 
    dateOfIssue.isNotEmpty || 
    dateOfExpiry.isNotEmpty;
}

class Detection {
  final double confidence;
  final int classId;
  final String className;
  final double x1, y1, x2, y2; // Bounding box coordinates
  Uint8List? croppedImage; // Store the cropped image bytes
  String? ocrText; // Store OCR results
  ParsedCnicData? parsedData; // Store parsed CNIC data

  Detection({
    required this.confidence,
    required this.classId,
    required this.className,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    this.croppedImage,
    this.ocrText,
    this.parsedData,
  });
}

class CnicDetectionService {
  Interpreter? _interpreter;
  static const int inputSize = 640;
  static const int maxDetections = 300;
  static const int numClasses = 12;
  static const int outputChannels = 6;
  
  // Class names based on your updated model
  final List<String> classNames = [
    'cnic',
    'cnic_back',
    'cnic_back_number',
    'cnic_left',
    'cnic_left_back',
    'cnic_left_back_number',
    'cnic_right',
    'cnic_right_back',
    'cnic_right_back_number',
    'cnic_upside_down',
    'cnic_upside_down_back',
    'cnic_upside_down_back_number'
  ];

  Future<bool> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/best_float32.tflite');
      print('Model loaded successfully');
      print('Input tensors: ${_interpreter!.getInputTensors()}');
      print('Output tensors: ${_interpreter!.getOutputTensors()}');
      
      // Verify the shapes match expectations
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      print('Input shape: $inputShape');
      print('Output shape: $outputShape');
      
      return true;
    } catch (e) {
      print('Error loading model: $e');
      return false;
    }
  }

  List<Detection> detectObjects(img.Image image) {
    if (_interpreter == null) {
      throw Exception('Model not loaded');
    }

    // Preprocess image
    final inputData = _preprocessImage(image);
    
    // Prepare input tensor - TFLite expects a 4D tensor [1, 640, 640, 3]
    final input = inputData.buffer.asFloat32List().reshape([1, inputSize, inputSize, 3]);
    
    // Prepare output buffer with correct shape [1, 300, 6]
    final output = List.generate(1, (_) => 
      List.generate(maxDetections, (_) => 
        List.generate(outputChannels, (_) => 0.0)
      )
    );
    
    // Run inference
    _interpreter!.run(input, output);
    
    // Post-process results
    final detections = _postProcessOutput(output[0]);
    
    // Extract and rotate cropped regions based on class name
    _extractCroppedRegions(image, detections);
    
    return detections;
  }

  Float32List _preprocessImage(img.Image image) {
    // Resize image maintaining aspect ratio
    final resized = img.copyResize(image, width: inputSize, height: inputSize);
    
    // Convert to float32 and normalize to [0,1]
    final input = Float32List(1 * inputSize * inputSize * 3);
    int pixelIndex = 0;
    
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        // RGB order and normalize to [0,1]
        input[pixelIndex++] = pixel.r / 255.0;
        input[pixelIndex++] = pixel.g / 255.0;
        input[pixelIndex++] = pixel.b / 255.0;
      }
    }
    
    return input;
  }

  List<Detection> _postProcessOutput(List<List<double>> output) {
    List<Detection> detections = [];
    const double confidenceThreshold = 0.5;
    
    // Output format: [300, 6] where each detection is [x1, y1, x2, y2, confidence, class_id]
    for (int i = 0; i < maxDetections; i++) {
      final detection = output[i];
      
      // Extract values from the detection
      final double x1 = detection[0];
      final double y1 = detection[1]; 
      final double x2 = detection[2];
      final double y2 = detection[3];
      final double confidence = detection[4];
      final int classId = detection[5].round();
      
      // Filter by confidence threshold
      if (confidence > confidenceThreshold && classId >= 0 && classId < classNames.length) {
        detections.add(Detection(
          confidence: confidence,
          classId: classId,
          className: classNames[classId],
          x1: x1, y1: y1, x2: x2, y2: y2,
        ));
        
        print('Detection $i: [${x1.toStringAsFixed(3)}, ${y1.toStringAsFixed(3)}, ${x2.toStringAsFixed(3)}, ${y2.toStringAsFixed(3)}]');
        print('Detection $i: Class=${classNames[classId]}, Confidence=${confidence.toStringAsFixed(2)}');
      }
    }
    
    // Sort by confidence (highest first)
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));
    
    print('Total detections after filtering: ${detections.length}');
    return detections;
  }

  void _extractCroppedRegions(img.Image originalImage, List<Detection> detections) {
    for (int i = 0; i < detections.length; i++) {
      final detection = detections[i];
      
      // Convert normalized coordinates to pixel coordinates
      int x1 = (detection.x1 * originalImage.width).round().clamp(0, originalImage.width - 1);
      int y1 = (detection.y1 * originalImage.height).round().clamp(0, originalImage.height - 1);
      int x2 = (detection.x2 * originalImage.width).round().clamp(0, originalImage.width);
      int y2 = (detection.y2 * originalImage.height).round().clamp(0, originalImage.height);
      
      // Ensure valid bounding box
      int width = (x2 - x1).abs();
      int height = (y2 - y1).abs();
      
      if (width > 0 && height > 0) {
        try {
          // Crop the image
          img.Image croppedImage = img.copyCrop(
            originalImage,
            x: x1,
            y: y1,
            width: width,
            height: height,
          );
          
          // Rotate based on class name
          if (detection.className.contains('right')) {
            croppedImage = img.copyRotate(croppedImage, angle: -90);
          } else if (detection.className.contains('left')) {
            croppedImage = img.copyRotate(croppedImage, angle: 90);
          } else if (detection.className.contains('upside_down')) {
            croppedImage = img.copyRotate(croppedImage, angle: 180);
          }
          
          // Convert to bytes
          final croppedBytes = Uint8List.fromList(img.encodePng(croppedImage));
          detection.croppedImage = croppedBytes;
          
          print('Cropped region ${i + 1}: ${width}x${height} pixels');
        } catch (e) {
          print('Error cropping region ${i + 1}: $e');
        }
      }
    }
  }

  Future<String?> performOcr(Uint8List imageBytes) async {
    try {
      // Save image temporarily for OCR processing
      final tempDir = await getTemporaryDirectory();
      final tempFile = File('${tempDir.path}/temp_ocr_image.png');
      await tempFile.writeAsBytes(imageBytes);

      // Initialize Google ML Kit Text Recognizer
      final textRecognizer = TextRecognizer(script: TextRecognitionScript.latin);
      final inputImage = InputImage.fromFilePath(tempFile.path);
      
      // Process image
      final RecognizedText recognizedText = await textRecognizer.processImage(inputImage);
      
      // Clean up
      await tempFile.delete();
      textRecognizer.close();
      
      return recognizedText.text;
    } catch (e) {
      print('OCR Error: $e');
      return null;
    }
  }

  ParsedCnicData parseCnicData(String ocrText) {
    if (ocrText.isEmpty) return ParsedCnicData();

    String name = '';
    String fatherName = '';
    String gender = '';
    String countryOfStay = '';
    String identityNumber = '';
    String dateOfBirth = '';
    String dateOfIssue = '';
    String dateOfExpiry = '';

    // Split OCR text into lines for more reliable parsing
    final lines = ocrText.split('\n').map((line) => line.trim()).where((line) => line.isNotEmpty).toList();
    
    // Extract Name (next line after "Name")
    for (int i = 0; i < lines.length; i++) {
      if ((lines[i].toLowerCase().contains('name') || (lines[i].toLowerCase().contains('nane') || (lines[i].toLowerCase().contains('narne')) && !lines[i].toLowerCase().contains('father')))) {
        if (i + 1 < lines.length) {
          name = lines[i + 1].trim();
          name = name.replaceAll(RegExp(r'[^\w\s]'), '').trim();
          if (name.length < 3 || name.length > 50) name = '';
        }
        break;
      }
    }

    // Extract Father Name (next line after "Father Name")
    for (int i = 0; i < lines.length; i++) {
      if (lines[i].toLowerCase().contains('father')) {
        if (i + 1 < lines.length) {
          fatherName = lines[i + 1].trim();
          fatherName = fatherName.replaceAll(RegExp(r'[^\w\s]'), '').trim();
          if (fatherName.length < 3 || fatherName.length > 50) fatherName = '';
        }
        break;
      }
    }

    // Extract Gender (M or F)
    final genderPattern = RegExp(r'\b([MF])\b', caseSensitive: false);
    for (final line in lines) {
      final match = genderPattern.firstMatch(line);
      if (match != null && match.group(1) != null) {
        gender = match.group(1)!.toUpperCase();
        if (gender == 'M' || gender == 'F') break;
      }
    }

    // Extract Country of Stay
    for (final line in lines) {
      if (line.toLowerCase().contains('pakistan')) {
        countryOfStay = 'Pakistan';
        break;
      }
    }

    // Extract Identity Number
    final idPattern = RegExp(r'(\d{5}-\d{7}-\d)');
    for (final line in lines) {
      final match = idPattern.firstMatch(line);
      if (match != null && match.group(1) != null) {
        identityNumber = match.group(1)!;
        break;
      }
    }

    // Extract and categorize dates
    final datePattern = RegExp(r'(\d{2}\.\d{2}\.\d{4})');
    final dates = <String>[];
    for (final line in lines) {
      final matches = datePattern.allMatches(line);
      for (final match in matches) {
        if (match.group(1) != null) {
          dates.add(match.group(1)!);
        }
      }
    }

    // Sort dates by year
    if (dates.isNotEmpty) {
      final dateWithYears = dates.map((date) {
        final parts = date.split('.');
        return {'date': date, 'year': int.parse(parts[2])};
      }).toList();
      
      dateWithYears.sort((a, b) => (a['year'] as int).compareTo(b['year'] as int));
      
      // Assign dates based on rules: earliest is birth, middle is issue, latest is expiry
      if (dateWithYears.length >= 3) {
        dateOfBirth = dateWithYears[0]['date'] as String;
        dateOfIssue = dateWithYears[1]['date'] as String;
        dateOfExpiry = dateWithYears[2]['date'] as String;
      } else if (dateWithYears.length == 2) {
        dateOfBirth = dateWithYears[0]['date'] as String;
        dateOfIssue = dateWithYears[0]['date'] as String;
        // Calculate expiry (10 years after issue)
        final issueYear = int.parse(dateOfIssue.split('.')[2]);
        dateOfExpiry = dateOfIssue.replaceAll(RegExp(r'\d{4}$'), '${issueYear + 10}');
      } else if (dateWithYears.length == 1) {
        dateOfBirth = dateWithYears[0]['date'] as String;
        dateOfIssue = dateWithYears[0]['date'] as String;
        final issueYear = int.parse(dateOfIssue.split('.')[2]);
        dateOfExpiry = dateOfIssue.replaceAll(RegExp(r'\d{4}$'), '${issueYear + 10}');
      }
    }

    return ParsedCnicData(
      name: name,
      fatherName: fatherName,
      gender: gender,
      countryOfStay: countryOfStay,
      identityNumber: identityNumber,
      dateOfBirth: dateOfBirth,
      dateOfIssue: dateOfIssue,
      dateOfExpiry: dateOfExpiry,
    );
  }

  bool compareIdentityNumbers(List<Detection> frontDetections, List<Detection> backDetections) {
    String? frontId;
    String? backId;

    for (var detection in frontDetections) {
      if (detection.parsedData?.identityNumber.isNotEmpty ?? false) {
        frontId = detection.parsedData!.identityNumber;
        break;
      }
    }

    for (var detection in backDetections) {
      if (detection.parsedData?.identityNumber.isNotEmpty ?? false) {
        backId = detection.parsedData!.identityNumber;
        break;
      }
    }

    if (frontId == null || backId == null) return false;
    return frontId == backId;
  }

  void dispose() {
    _interpreter?.close();
  }
}

// Custom Painter for drawing bounding boxes
class BoundingBoxPainter extends CustomPainter {
  final List<Detection> detections;
  final Size imageSize; // Original image dimensions
  final Size displaySize; // Widget display dimensions
  final Offset imageOffset; // Offset due to BoxFit.contain

  BoundingBoxPainter({
    required this.detections,
    required this.imageSize,
    required this.displaySize,
    required this.imageOffset,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    // Calculate scale factor from original image to display
    final scaleX = displaySize.width / imageSize.width;
    final scaleY = displaySize.height / imageSize.height;

    for (final detection in detections) {
      // Transform normalized coordinates (0-1) to original image coordinates
      final originalX1 = detection.x1 * imageSize.width;
      final originalY1 = detection.y1 * imageSize.height;
      final originalX2 = detection.x2 * imageSize.width;
      final originalY2 = detection.y2 * imageSize.height;

      // Transform to display coordinates
      final displayX1 = originalX1 * scaleX;
      final displayY1 = originalY1 * scaleY + imageOffset.dy;
      final displayX2 = originalX2 * scaleX;
      final displayY2 = originalY2 * scaleY + imageOffset.dy;

      // Draw bounding box
      final rect = Rect.fromLTRB(displayX1, displayY1, displayX2, displayY2);
      canvas.drawRect(rect, paint);
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}

class DisplayInfo {
  final Size displaySize;
  final Offset offset;

  DisplayInfo({required this.displaySize, required this.offset});
}

class CameraScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  
  const CameraScreen({Key? key, required this.cameras}) : super(key: key);

  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _cameraController;
  CnicDetectionService? _detectionService;
  bool _isModelLoaded = false;
  bool _isCapturing = false;
  bool _isCapturingFront = true; // Track whether capturing front or back
  Uint8List? _frontImageBytes;
  List<Detection> _frontDetections = [];
  Uint8List? _backImageBytes;
  List<Detection> _backDetections = [];

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializeModel();
    // Lock orientation to portrait for consistency
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
    ]);
  }

  late int _sensorOrientation;

  Future<void> _initializeCamera() async {
    if (widget.cameras.isEmpty) {
      _showError('No cameras available.');
      return;
    }
    
    await _requestCameraPermission();
    
    _cameraController = CameraController(
      widget.cameras[0],
      ResolutionPreset.veryHigh,
      enableAudio: false,
    );
    
    _sensorOrientation = widget.cameras[0].sensorOrientation;
    print('Camera sensor orientation: $_sensorOrientation');

    try {
      await _cameraController!.initialize();
      setState(() {});
    } catch (e) {
      _showError('Error initializing camera: $e');
      print('Error initializing camera: $e');
    }
  }

  Future<void> _requestCameraPermission() async {
    final status = await Permission.camera.request();
    if (status != PermissionStatus.granted) {
      _showError('Camera permission not granted.');
      throw Exception('Camera permission not granted');
    }
  }

  Future<void> _initializeModel() async {
    _detectionService = CnicDetectionService();
    final loaded = await _detectionService!.loadModel();
    setState(() {
      _isModelLoaded = loaded;
    });
  }

  Future<void> _captureAndDetect() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized || _isCapturing || _detectionService == null) {
      _showError('Camera or model not initialized.');
      return;
    }

    setState(() {
      _isCapturing = true;
    });

    try {
      final XFile imageFile = await _cameraController!.takePicture();
      final File file = File(imageFile.path);
      final Uint8List imageBytes = await file.readAsBytes();
      
      // Decode image
      final img.Image? image = img.decodeImage(imageBytes);
      if (image == null) {
        throw Exception('Failed to decode image');
      }

      // Run detection
      final detections = _detectionService!.detectObjects(image);
      
      if (_isCapturingFront) {
        // Validate front capture: no classes should contain "back"
        if (detections.isEmpty) {
          _showError('No CNIC detected. Please capture the front of the CNIC.');
        } else if (detections.any((d) => d.className.toLowerCase().contains('back'))) {
          _showError('Detected back of CNIC. Please capture the front of the CNIC.');
        } else {
          // Front capture successful
          setState(() {
            _frontImageBytes = imageBytes;
            _frontDetections = detections;
            _isCapturingFront = false;
          });
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Front captured successfully. Now capture the back of the CNIC.'),
              backgroundColor: Colors.green,
            ),
          );
        }
      } else {
        // Validate back capture: all classes must contain "back"
        if (detections.isEmpty) {
          _showError('No CNIC detected. Please capture the back of the CNIC.');
        } else if (detections.any((d) => !d.className.toLowerCase().contains('back'))) {
          _showError('Detected front of CNIC. Please capture the back of the CNIC.');
        } else {
          // Back capture successful, perform OCR
          _backImageBytes = imageBytes;
          _backDetections = detections;

          // Perform OCR on front (non-back classes) and back (number classes)
          for (var detection in _frontDetections) {
            if (detection.croppedImage != null && !detection.className.toLowerCase().contains('back')) {
              final ocrResult = await _detectionService!.performOcr(detection.croppedImage!);
              detection.ocrText = ocrResult ?? 'No text detected';
              if (ocrResult != null && ocrResult.isNotEmpty) {
                detection.parsedData = _detectionService!.parseCnicData(ocrResult);
              }
            }
          }
          for (var detection in _backDetections) {
            if (detection.croppedImage != null && detection.className.toLowerCase().contains('number')) {
              final ocrResult = await _detectionService!.performOcr(detection.croppedImage!);
              detection.ocrText = ocrResult ?? 'No text detected';
              if (ocrResult != null && ocrResult.isNotEmpty) {
                detection.parsedData = _detectionService!.parseCnicData(ocrResult);
              }
            }
          }

          // Combine detections
          final allDetections = [..._frontDetections, ..._backDetections];

          // Check if images are available before navigating
          if (_frontImageBytes == null || _backImageBytes == null) {
            _showError('Error: Missing captured images. Please try again.');
            setState(() {
              _isCapturingFront = true;
              _frontImageBytes = null;
              _frontDetections = [];
              _backImageBytes = null;
              _backDetections = [];
            });
            return;
          }

          // Navigate to results page
          await Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => ResultsScreen(
                frontImageBytes: _frontImageBytes!,
                backImageBytes: _backImageBytes!,
                detections: allDetections,
                detectionService: _detectionService!,
              ),
            ),
          );

          // Reset state to capture front again after navigation
          setState(() {
            _isCapturingFront = true;
            _frontImageBytes = null;
            _frontDetections = [];
            _backImageBytes = null;
            _backDetections = [];
          });
        }
      }
    } catch (e) {
      _showError('Error: $e');
      print('Capture error: $e');
    } finally {
      setState(() {
        _isCapturing = false;
      });
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Scaffold(
        backgroundColor: Colors.black,
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(color: Colors.white),
              SizedBox(height: 16),
              Text(
                'Initializing Camera...',
                style: TextStyle(color: Colors.white, fontSize: 16),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Camera preview as full background
          Center(child: CameraPreview(_cameraController!)),
          
          // Column layout for slabs
          Column(
            children: [
              // Top slab
              Container(
                height: 120,
                color: Colors.black,
                child: SafeArea(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    child: Row(
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Text(
                                'CNIC Detection',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Row(
                                children: [
                                  Icon(
                                    _isModelLoaded ? Icons.check_circle : Icons.hourglass_empty,
                                    color: _isModelLoaded ? Colors.green : Colors.orange,
                                    size: 12,
                                  ),
                                  const SizedBox(width: 6),
                                  Text(
                                    _isModelLoaded ? 'Ready' : 'Loading...',
                                    style: TextStyle(
                                      color: Colors.white70,
                                      fontSize: 12,
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              
              // Expanded camera area
              Expanded(child: Container()),
              
              // Bottom slab with capture button
              Container(
                height: 90,
                color: Colors.black,
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        _isCapturingFront
                            ? 'Capture the front of the CNIC'
                            : 'Capture the back of the CNIC',
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.7),
                          fontSize: 11,
                        ),
                      ),
                      const SizedBox(height: 8),
                      GestureDetector(
                        onTap: _isModelLoaded && !_isCapturing ? _captureAndDetect : null,
                        child: AnimatedContainer(
                          duration: Duration(milliseconds: 200),
                          width: 55,
                          height: 55,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: _isModelLoaded 
                                ? (_isCapturing ? Colors.grey[400] : Colors.white)
                                : Colors.grey.withOpacity(0.5),
                            border: Border.all(
                              color: _isModelLoaded ? Colors.white : Colors.grey,
                              width: 2,
                            ),
                            boxShadow: _isModelLoaded ? [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.3),
                                spreadRadius: 1,
                                blurRadius: 4,
                                offset: Offset(0, 1),
                              ),
                            ] : null,
                          ),
                          child: _isCapturing
                              ? const CircularProgressIndicator(
                                  valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
                                  strokeWidth: 2,
                                )
                              : Container(
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    color: _isModelLoaded ? Colors.red : Colors.grey[600],
                                  ),
                                  margin: EdgeInsets.all(6),
                                ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _detectionService?.dispose();
    super.dispose();
  }
}

class ResultsScreen extends StatelessWidget {
  final Uint8List frontImageBytes;
  final Uint8List backImageBytes;
  final List<Detection> detections;
  final CnicDetectionService detectionService;

  const ResultsScreen({
    Key? key,
    required this.frontImageBytes,
    required this.backImageBytes,
    required this.detections,
    required this.detectionService,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final frontDetections = detections.where((d) => !d.className.toLowerCase().contains('back')).toList();
    final backDetections = detections.where((d) => d.className.toLowerCase().contains('back')).toList();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Captured Images'),
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      backgroundColor: Colors.black,
      body: Column(
        children: [
          // Image display with bounding boxes
          Expanded(
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(8),
              child: DefaultTabController(
                length: 2,
                child: Column(
                  children: [
                    TabBar(
                      labelColor: Colors.white,
                      unselectedLabelColor: Colors.grey[400],
                      indicatorColor: Colors.blue,
                      tabs: const [
                        Tab(text: 'Front'),
                        Tab(text: 'Back'),
                      ],
                    ),
                    Expanded(
                      child: TabBarView(
                        children: [
                          _buildImageView(context, frontImageBytes, frontDetections),
                          _buildImageView(context, backImageBytes, backDetections),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          // Bottom slab with See Info button
          Container(
            height: 90,
            color: Colors.black,
            child: Center(
              child: ElevatedButton.icon(
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => DataScreen(
                      frontDetections: frontDetections,
                      backDetections: backDetections,
                      detectionService: detectionService,
                    ),
                  ),
                ),
                icon: const Icon(Icons.info),
                label: const Text('See Info'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageView(BuildContext context, Uint8List imageBytes, List<Detection> detections) {
    return InteractiveViewer(
      panEnabled: true,
      scaleEnabled: true,
      minScale: 0.5,
      maxScale: 4.0,
      child: Center(
        child: LayoutBuilder(
          builder: (context, constraints) {
            return FutureBuilder<img.Image?>(
              future: _decodeImage(imageBytes),
              builder: (context, snapshot) {
                if (!snapshot.hasData || snapshot.data == null) {
                  return const CircularProgressIndicator();
                }

                final image = snapshot.data!;
                final imageSize = Size(image.width.toDouble(), image.height.toDouble());
                
                // Calculate display size and offset for BoxFit.contain
                final containerSize = Size(constraints.maxWidth, constraints.maxHeight);
                final displayInfo = _calculateDisplayInfo(imageSize, containerSize);

                return Stack(
                  children: [
                    // Original image
                    Image.memory(
                      imageBytes,
                      fit: BoxFit.contain,
                    ),
                    // Bounding boxes overlay
                    if (detections.isNotEmpty)
                      Positioned.fill(
                        child: CustomPaint(
                          painter: BoundingBoxPainter(
                            detections: detections,
                            imageSize: imageSize,
                            displaySize: displayInfo.displaySize,
                            imageOffset: displayInfo.offset,
                          ),
                        ),
                      ),
                  ],
                );
              },
            );
          },
        ),
      ),
    );
  }

  Future<img.Image?> _decodeImage(Uint8List bytes) async {
    return img.decodeImage(bytes);
  }

  DisplayInfo _calculateDisplayInfo(Size imageSize, Size containerSize) {
    // Calculate how BoxFit.contain displays the image
    final imageAspectRatio = imageSize.width / imageSize.height;
    final containerAspectRatio = containerSize.width / containerSize.height;

    late Size displaySize;
    late Offset offset;

    if (imageAspectRatio > containerAspectRatio) {
      // Image is wider - limited by width
      displaySize = Size(
        containerSize.width,
        containerSize.width / imageAspectRatio,
      );
      offset = Offset(0, (containerSize.height - displaySize.height) / 2);
    } else {
      // Image is taller - limited by height
      displaySize = Size(
        containerSize.height * imageAspectRatio,
        containerSize.height,
      );
      offset = Offset((containerSize.width - displaySize.width) / 2, 0);
    }

    return DisplayInfo(displaySize: displaySize, offset: offset);
  }
}

class DataScreen extends StatelessWidget {
  final List<Detection> frontDetections;
  final List<Detection> backDetections;
  final CnicDetectionService detectionService;

  const DataScreen({
    Key? key,
    required this.frontDetections,
    required this.backDetections,
    required this.detectionService,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final isValidCnic = detectionService.compareIdentityNumbers(frontDetections, backDetections);
    ParsedCnicData? parsedData;
    List<Detection> validDetections = [];

    if (isValidCnic) {
      // Use front detection data for display (preferring front for completeness)
      for (var detection in frontDetections) {
        if (detection.parsedData?.hasAnyData ?? false) {
          parsedData = detection.parsedData;
          if (detection.ocrText != null && detection.ocrText!.isNotEmpty && detection.ocrText != 'No text detected') {
            validDetections.add(detection);
          }
        }
      }
      // Include back detections with number classes
      for (var detection in backDetections) {
        if (detection.className.toLowerCase().contains('number') && (detection.parsedData?.hasAnyData ?? false)) {
          if (detection.ocrText != null && detection.ocrText!.isNotEmpty && detection.ocrText != 'No text detected') {
            validDetections.add(detection);
          }
        }
      }
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Extracted Data'),
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      backgroundColor: Colors.black,
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: isValidCnic && parsedData != null
            ? SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'CNIC Details',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    _buildInfoRow('Identity Number:', parsedData.identityNumber),
                    _buildInfoRow('Name:', parsedData.name),
                    _buildInfoRow('Father Name:', parsedData.fatherName),
                    _buildInfoRow('Gender:', parsedData.gender),
                    _buildInfoRow('Country of Stay:', parsedData.countryOfStay),
                    _buildInfoRow('Date of Birth:', parsedData.dateOfBirth),
                    _buildInfoRow('Date of Issue:', parsedData.dateOfIssue),
                    _buildInfoRow('Date of Expiry:', parsedData.dateOfExpiry),
                  ],
                ),
              )
            : Center(
                child: Text(
                  'Invalid CNIC. Please try again!',
                  style: TextStyle(
                    color: Colors.red,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 120,
            child: Text(
              label,
              style: TextStyle(
                color: Colors.grey[400],
                fontSize: 14,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value.isEmpty ? 'Not detected' : value,
              style: TextStyle(
                color: value.isEmpty ? Colors.grey[600] : Colors.white,
                fontSize: 14,
              ),
            ),
          ),
        ],
      ),
    );
  }
}