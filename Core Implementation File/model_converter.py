import tensorflow as tf
import argparse
import os

def convert_to_tflite(saved_model_dir, output_file, quantize=True):
    """
    Convert saved model to TFLite format with optional quantization.
    
    Args:
        saved_model_dir: Directory containing SavedModel
        output_file: Output TFLite file path
        quantize: Whether to apply int8 quantization
    """
    print(f"Loading model from {saved_model_dir}")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    
    if quantize:
        print("Applying int8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Define representative dataset generator
        def representative_dataset_gen():
            # In practice, this should load calibration data
            for _ in range(100):
                yield [np.random.uniform(0, 1, (1, 240, 240, 3)).astype(np.float32)]
        
        converter.representative_dataset = representative_dataset_gen
    
    print("Converting model...")
    tflite_model = converter.convert()
    
    # Save model
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted and saved to {output_file}")
    print(f"Model size: {os.path.getsize(output_file) / 1024:.2f} KB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert TensorFlow model to TFLite format')
    parser.add_argument('--saved_model', required=True, help='Path to SavedModel directory')
    parser.add_argument('--output', required=True, help='Output TFLite model path')
    parser.add_argument('--quantize', action='store_true', help='Apply int8 quantization')
    
    args = parser.parse_args()
    convert_to_tflite(args.saved_model, args.output, args.quantize)
