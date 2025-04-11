# This script demonstrates the edge-optimized object detection system using a webcam or video file as input.

import argparse
import cv2
import time
import numpy as np
from src.detect import ObjectDetector
from src.visualization import draw_detections

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Edge AI Object Detection Demo')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--camera', type=int, help='Camera device ID')
    group.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, required=True, help='Path to TFLite model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--output', type=str, help='Output video file path')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Initialize detector
    detector = ObjectDetector(args.model, args.threshold)
    
    # Initialize video capture
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = cv2.VideoCapture(args.video)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output is specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Performance metrics
    frame_count = 0
    total_inference_time = 0
    start_time = time.time()
    
    # Main loop
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        inference_start = time.time()
        detections = detector.detect(frame)
        inference_time = time.time() - inference_start
        
        # Update metrics
        frame_count += 1
        total_inference_time += inference_time
        
        # Draw detections
        result_frame = draw_detections(frame, detections)
        
        # Add performance metrics to frame
        avg_inference_time = total_inference_time / frame_count
        fps_value = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        
        cv2.putText(result_frame, f"Inference: {inference_time*1000:.1f}ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result_frame, f"FPS: {fps_value:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Write frame to output if specified
        if writer:
            writer.write(result_frame)
        
        # Display frame
        cv2.imshow("Edge AI Object Detection", result_frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Calculate and display overall statistics
    elapsed_time = time.time() - start_time
    print(f"\nPerformance Statistics:")
    print(f"Total frames: {frame_count}")
    print(f"Average inference time: {total_inference_time/frame_count*1000:.2f}ms")
    print(f"Average FPS: {frame_count/elapsed_time:.2f}")
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
