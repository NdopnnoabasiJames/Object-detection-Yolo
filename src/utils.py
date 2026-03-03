"""
YOLO Object Detection - Utility Functions
Helper functions for data processing, visualization, and model utilities
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(image_path):
    """
    Load an image from path
    
    Args:
        image_path: Path to the image
        
    Returns:
        Loaded image
    """
    image = cv2.imread(str(image_path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(image, output_path):
    """
    Save an image to path
    
    Args:
        image: Image array
        output_path: Path to save the image
    """
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)


def draw_bounding_boxes(image, detections, class_names=None):
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image
        detections: List of detections (x1, y1, x2, y2, conf, class_id)
        class_names: List of class names
        
    Returns:
        Image with bounding boxes
    """
    # TODO: Implement bounding box drawing
    pass


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    
    Args:
        box1: First box (x1, y1, x2, y2)
        box2: Second box (x1, y1, x2, y2)
        
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def non_max_suppression(detections, iou_threshold=0.45):
    """
    Apply Non-Maximum Suppression
    
    Args:
        detections: List of detections
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Filtered detections
    """
    # TODO: Implement NMS
    pass


def plot_training_metrics(metrics, save_path=None):
    """
    Plot training metrics
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the plot
    """
    # TODO: Implement metrics plotting
    pass


if __name__ == "__main__":
    print("YOLO Utility Functions")
