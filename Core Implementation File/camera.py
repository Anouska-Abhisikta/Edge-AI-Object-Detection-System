import cv2

class Camera:
    """Camera interface for video capture."""
    
    def __init__(self, device_id=0, width=640, height=480):
        """
        Initialize camera capture.
        
        Args:
            device_id: Camera device ID
            width: Capture width
            height: Capture height
        """
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {device_id}")
            
        # Warm up camera
        for _ in range(5):
            self.cap.read()
    
    def read(self):
        """Read a frame from camera."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image")
        return frame
    
    def release(self):
        """Release camera resources."""
        self.cap.release()

# src/visualization.py
import cv2
import numpy as np
from typing import List
from .detect import Detection

def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """
    Draw detection results on the image.
    
    Args:
        image: Input image
        detections: List of Detection objects
        
    Returns:
        Image with drawn detections
    """
    # Create a copy of the image
    output = image.copy()
    
    # Define colors for different classes (for simplicity, generate from class id)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    for detection in detections:
        # Get bounding box coordinates (convert to integers)
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        
        # Choose color based on label hash
        color_idx = abs(hash(detection.label)) % len(colors)
        color = colors[color_idx]
        
        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f"{detection.label}: {detection.confidence:.2f}"
        cv2.putText(output, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return output

