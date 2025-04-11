import cv2
import numpy as np

def resize_and_pad(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize image keeping aspect ratio and pad with zeros.
    
    Args:
        image: Input image
        target_size: Target size (width, height)
        
    Returns:
        Resized and padded image
    """
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Calculate target dimensions
    tw, th = target_size
    
    # Calculate scaling factor
    scale = min(tw / w, th / h)
    
    # Calculate new dimensions
    nw, nh = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (nw, nh))
    
    # Create target image (zero padding)
    target = np.zeros((th, tw, 3), dtype=np.uint8)
    
    # Calculate padding
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    
    # Copy resized image to target
    target[dy:dy+nh, dx:dx+nw] = resized
    
    return target

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0

