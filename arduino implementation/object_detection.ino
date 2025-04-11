#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h" // Model weights and parameters header file

// Camera library depends on the specific camera module being used
#if defined(ARDUINO_PORTENTA_H7)
#include "Camera.h"
#elif defined(ARDUINO_ARDUINO_NANO33BLE)
#include <Arduino_OV767X.h>
#endif

// Globals for TensorFlow Lite
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// Memory allocation for TensorFlow Lite arena
constexpr int kTensorArenaSize = 136 * 1024; // Adjust based on model requirements
uint8_t tensor_arena[kTensorArenaSize];

// Camera buffer
#define IMAGE_WIDTH 240
#define IMAGE_HEIGHT 240
#define CHANNELS 3
uint8_t image_buffer[IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS];

// Detection parameters
constexpr float DETECTION_THRESHOLD = 0.5f;
constexpr int NUM_CLASSES = 15;
constexpr int MAX_DETECTIONS = 10;

// Result arrays
float detection_boxes[MAX_DETECTIONS * 4]; // (y1, x1, y2, x2) normalized coordinates
float detection_scores[MAX_DETECTIONS];
float detection_classes[MAX_DETECTIONS];

// Class labels (same as Python implementation)
const char* labels[] = {
  "background", "person", "bicycle", "car", "motorcycle", 
  "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench"
};

void setup() {
  // Initialize serial for debugging
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  Serial.println("Edge AI Object Detection");

  // Setup LED indicators if available
  pinMode(LED_BUILTIN, OUTPUT);

  // Initialize camera
  if (!initCamera()) {
    Serial.println("Camera initialization failed!");
    while (1) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(100);
      digitalWrite(LED_BUILTIN, LOW);
      delay(100);
    }
  }
  Serial.println("Camera initialized");

  // Initialize TensorFlow Lite model
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(500);
      digitalWrite(LED_BUILTIN, LOW);
      delay(500);
    }
  }

  // Create TensorFlow Lite interpreter
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(1000);
      digitalWrite(LED_BUILTIN, LOW);
      delay(1000);
    }
  }

  // Get input tensor
  input = interpreter->input(0);
  
  // Check input dimensions match our buffer
  if (input->dims->size != 4 || 
      input->dims->data[1] != IMAGE_HEIGHT || 
      input->dims->data[2] != IMAGE_WIDTH || 
      input->dims->data[3] != CHANNELS) {
    Serial.println("Input tensor dimensions mismatch!");
    Serial.print("Expected: [1, ");
    Serial.print(IMAGE_HEIGHT);
    Serial.print(", ");
    Serial.print(IMAGE_WIDTH);
    Serial.print(", ");
    Serial.print(CHANNELS);
    Serial.println("]");
    Serial.print("Got: [");
    for (int i = 0; i < input->dims->size; i++) {
      Serial.print(input->dims->data[i]);
      if (i < input->dims->size - 1) Serial.print(", ");
    }
    Serial.println("]");
    while (1) {
      digitalWrite(LED_BUILTIN, HIGH);
      delay(300);
      digitalWrite(LED_BUILTIN, LOW);
      delay(300);
    }
  }

  Serial.println("Model initialized and ready!");
}

void loop() {
  // Capture image
  if (!captureImage(image_buffer)) {
    Serial.println("Failed to capture image!");
    delay(1000);
    return;
  }

  // Signal processing start
  digitalWrite(LED_BUILTIN, HIGH);
  
  // Preprocess image into input tensor
  preprocessImage(image_buffer, input);
  
  // Run inference
  unsigned long start_time = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long inference_time = millis() - start_time;
  
  // Signal processing end
  digitalWrite(LED_BUILTIN, LOW);
  
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    delay(1000);
    return;
  }

  Serial.print("Inference time: ");
  Serial.print(inference_time);
  Serial.println(" ms");
  
  // Get output tensors
  TfLiteTensor* output_boxes = interpreter->output(0);
  TfLiteTensor* output_classes = interpreter->output(1);
  TfLiteTensor* output_scores = interpreter->output(2);
  TfLiteTensor* num_detections = interpreter->output(3);
  
  // Process results
  int num_boxes = static_cast<int>(num_detections->data.f[0]);
  for (int i = 0; i < min(num_boxes, MAX_DETECTIONS); i++) {
    detection_scores[i] = output_scores->data.f[i];
    if (detection_scores[i] >= DETECTION_THRESHOLD) {
      // Class index (add 1 if your model starts from 1)
      int class_id = static_cast<int>(output_classes->data.f[i]);
      detection_classes[i] = class_id;
      
      // Bounding box coordinates (y1, x1, y2, x2)
      for (int j = 0; j < 4; j++) {
        detection_boxes[i * 4 + j] = output_boxes->data.f[i * 4 + j];
      }
      
      // Print detection result
      Serial.print("Detection #");
      Serial.print(i);
      Serial.print(": ");
      Serial.print(labels[class_id]);
      Serial.print(" (score: ");
      Serial.print(detection_scores[i], 4);
      Serial.print("), box: [");
      Serial.print(detection_boxes[i * 4 + 1], 3); // x1
      Serial.print(", ");
      Serial.print(detection_boxes[i * 4 + 0], 3); // y1
      Serial.print(", ");
      Serial.print(detection_boxes[i * 4 + 3], 3); // x2
      Serial.print(", ");
      Serial.print(detection_boxes[i * 4 + 2], 3); // y2
      Serial.println("]");
    }
  }
  
  // Delay between detections
  delay(1000);
}

bool initCamera() {
  // Camera initialization code depends on the specific hardware
#if defined(ARDUINO_PORTENTA_H7)
  // Portenta H7 camera initialization
  if (!Camera.begin(QVGA, RGB888, 30)) {
    return false;
  }
#elif defined(ARDUINO_ARDUINO_NANO33BLE)
  // Nano 33 BLE camera initialization
  if (!Camera.begin(QVGA, RGB888, 30)) {
    return false;
  }
#else
  // Generic initialization - replace with your camera code
  Serial.println("WARNING: Using dummy camera, implement for your hardware");
  return true;
#endif
  return true;
}

bool captureImage(uint8_t* buffer) {
  // Camera capture code depends on the specific hardware
#if defined(ARDUINO_PORTENTA_H7)
  // Portenta H7 camera capture
  Camera.readFrame(buffer);
  return true;
#elif defined(ARDUINO_ARDUINO_NANO33BLE)
  // Nano 33 BLE camera capture
  Camera.readFrame(buffer);
  return true;
#else
  // Generic capture - replace with your camera code
  // For testing, fill with dummy data
  for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS; i++) {
    buffer[i] = random(256);
  }
  return true;
#endif
}

void preprocessImage(uint8_t* image_data, TfLiteTensor* input_tensor) {
  // Input normalization depends on model requirements

  // For int8 quantized model
  if (input_tensor->type == kTfLiteInt8) {
    int8_t* input_data_int8 = input_tensor->data.int8;
    
    // Get input quantization parameters
    float input_scale = input_tensor->params.scale;
    int input_zero_point = input_tensor->params.zero_point;
    
    for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS; i++) {
      // Convert uint8 [0, 255] to float [0, 1]
      float pixel_value = static_cast<float>(image_data[i]) / 255.0f;
      
      // Quantize float to int8
      int32_t quantized = static_cast<int32_t>(pixel_value / input_scale + input_zero_point);
      
      // Clamp to int8 range and store
      input_data_int8[i] = static_cast<int8_t>(
          min(127, max(-128, quantized)));
    }
  } 
  // For float model
  else if (input_tensor->type == kTfLiteFloat32) {
    float* input_data_float = input_tensor->data.f;
    
    for (int i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS; i++) {
      // Convert uint8 [0, 255] to float [0, 1]
      input_data_float[i] = static_cast<float>(image_data[i]) / 255.0f;
    }
  }
}

// model_data.h would be generated during the build process 
// and would contain the model weights and parameters as a byte array
