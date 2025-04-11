import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, NamedTuple
import tensorflow as tf

class Detection(NamedTuple):
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in normalized coordinates

class ObjectDetector:
    """Lightweight object detector using TensorFlow Lite for edge devices."""
    
    def __init__(self, model_path: str, threshold: float = 0.5):
        """
        Initialize the detector with a TFLite model.
        
        Args:
            model_path: Path to the TFLite model file
            threshold: Detection confidence threshold
        """
        self.threshold = threshold
        self.labels = self._load_labels()
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get model input shape
        self.input_shape = self.input_details[0]['shape'][1:3]
        print(f"Model loaded with input shape: {self.input_shape}")
        
    def _load_labels(self) -> List[str]:
        """Load the model labels."""
        # In a real implementation, load from a file
        return [
            "background", "person", "bicycle", "car", "motorcycle", 
            "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench"
        ]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for model input.
        
        Args:
            image: Input image in BGR format (OpenCV default)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Resize to model input size
        resized = cv2.resize(image, self.input_shape)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def detect(self, image_source) -> List[Detection]:
        """
        Detect objects in an image.
        
        Args:
            image_source: Path to image or numpy array
            
        Returns:
            List of Detection objects
        """
        # Handle different input types
        if isinstance(image_source, str):
            image = cv2.imread(image_source)
        else:
            image = image_source
            
        if image is None:
            raise ValueError("Invalid image source")
        
        # Save original dimensions for bbox scaling
        orig_height, orig_width = image.shape[:2]
        
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        
        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time*1000:.2f}ms")
        
        # Get detection results
        # Assuming output tensors are:
        # 1. boxes [1, num_boxes, 4] with values [y1, x1, y2, x2] normalized
        # 2. classes [1, num_boxes] with class indices
        # 3. scores [1, num_boxes] with confidence values
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        # Create detection list
        detections = []
        for i in range(len(scores)):
            if scores[i] > self.threshold:
                # Convert normalized box coordinates to pixel values
                y1, x1, y2, x2 = boxes[i]
                x1 = x1 * orig_width
                y1 = y1 * orig_height
                x2 = x2 * orig_width
                y2 = y2 * orig_height
                
                # Get class label
                class_id = int(classes[i])
                label = self.labels[class_id]
                
                # Add detection
                detections.append(Detection(
                    label=label,
                    confidence=float(scores[i]),
                    bbox=(x1, y1, x2, y2)
                ))
        
        return detections

