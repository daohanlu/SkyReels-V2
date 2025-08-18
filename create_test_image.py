#!/usr/bin/env python3
"""
Create a simple test image for video generation comparison.
"""

import numpy as np
from PIL import Image
import os

def create_test_image(output_path: str = "test_image.jpg", size: tuple = (960, 544)):
    """Create a simple test image with a gradient and some shapes."""
    width, height = size
    
    # Create a gradient background
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a sunset-like gradient
    for i in range(height):
        for j in range(width):
            # Red gradient from top to bottom
            red = int(255 * (1 - i / height))
            # Blue gradient from left to right
            blue = int(255 * (j / width))
            # Green constant
            green = 100
            
            img_array[i, j] = [red, green, blue]
    
    # Add some simple shapes
    # Add a circle in the center
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 8
    
    for i in range(height):
        for j in range(width):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if dist < radius:
                img_array[i, j] = [255, 255, 255]  # White circle
    
    # Add a rectangle
    rect_x1, rect_y1 = width // 4, height // 4
    rect_x2, rect_y2 = width // 4 + 100, height // 4 + 50
    img_array[rect_y1:rect_y2, rect_x1:rect_x2] = [255, 255, 0]  # Yellow rectangle
    
    # Create image and save
    image = Image.fromarray(img_array)
    image.save(output_path)
    
    print(f"✅ Test image created: {output_path}")
    print(f"   Size: {image.size}")
    print(f"   Mode: {image.mode}")
    
    return output_path

if __name__ == "__main__":
    # Create test image
    test_image_path = create_test_image()
    print(f"\nTest image ready for video generation: {test_image_path}")
