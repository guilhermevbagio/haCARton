import numpy as np
import cv2
from PIL import Image
import random

def mock_segmentation(image: np.ndarray, num_polygons: int = 5) -> list:
    """Mock segmentation: returns list of polygon masks."""
    h, w = image.shape[:2]
    polygons = []
    
    for _ in range(num_polygons):
        # Random center
        cx = random.randint(50, w - 50)
        cy = random.randint(50, h - 50)
        
        # Random polygon points
        num_points = random.randint(5, 8)
        points = []
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points
            r = random.randint(20, 60)
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))
            points.append([px, py])
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        polygons.append(mask)
    
    return polygons


def mock_classification(mask: np.ndarray) -> dict:
    """Mock classification: returns random class with confidence."""
    classes = [
        "forest", "urban", "water", "agriculture", "desert",
        "residential", "industrial", "wetland", "grassland"
    ]
    
    class_name = random.choice(classes)
    confidence = random.uniform(0.6, 0.99)
    
    return {
        "class": class_name,
        "confidence": round(confidence, 2)
    }


def process_image(image: np.ndarray) -> list:
    """Full mock pipeline: segment then classify."""
    polygons = mock_segmentation(image)
    results = []
    
    for i, mask in enumerate(polygons):
        classification = mock_classification(mask)
        results.append({
            "id": i,
            "mask": mask,
            "class": classification["class"],
            "confidence": classification["confidence"],
            "status": "pending"
        })
    
    return results