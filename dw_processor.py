import numpy as np
import tensorflow as tf
import os
import cv2

CLASS_NAMES = [
    "Water", "Trees", "Grass", "Flooded Vegetation", "Crops",
    "Scrub/Shrub", "Built Area", "Bare Ground", "Snow/Ice"
]

CLASS_COLORS = np.array([
    [41, 121, 185],    # Water - blue
    [56, 168, 73],     # Trees - green
    [163, 207, 55],    # Grass - light green
    [130, 185, 155],   # Flooded Vegetation - teal
    [235, 215, 80],    # Crops - yellow
    [193, 148, 66],    # Scrub/Shrub - brown
    [180, 60, 50],     # Built Area - red
    [160, 130, 90],    # Bare Ground - tan
    [230, 230, 240],   # Snow/Ice - white
], dtype=np.uint8)

NORM_PERCENTILES = np.array([
    [1.7417268007636313, 2.023298706048351],
    [1.7261204997060209, 2.038905204308012],
    [1.6798346251414997, 2.179592821212937],
    [1.7734969472909623, 2.2890068333026603],
    [2.289154079164943, 2.6171674549378166],
    [2.382939712192371, 2.773418590375327],
    [2.3828939530384052, 2.7578332604178284],
    [2.1952484264967844, 2.789092484314204],
    [1.554812948247501, 2.4140534947492487]
])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "dw_repo", "model", "forward")

_model = None

def get_model():
    global _model
    if _model is None:
        _model = tf.saved_model.load(MODEL_PATH)
    return _model


def normalize_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    rgb = image.astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    bands = np.stack([
        b * 2000, g * 2000, r * 2000,
        g * 1500, g * 1200, g * 1000,
        (r + g + b) / 3 * 1800,
        r * 500, b * 400
    ], axis=-1)

    image_log = np.log(bands * 0.005 + 1)
    image_norm = (image_log - NORM_PERCENTILES[:, 0]) / NORM_PERCENTILES[:, 1]
    image_sigmoid = np.exp(image_norm * 5 - 1)
    image_sigmoid = image_sigmoid / (image_sigmoid + 1)
    return image_sigmoid.astype(np.float32)


def classify_image(image: np.ndarray) -> dict:
    """Run Dynamic World classification and return results."""
    model = get_model()
    h, w = image.shape[:2]

    normalized = normalize_image(image)
    input_tensor = tf.constant(normalized[np.newaxis, ...])

    with tf.device('/CPU:0'):
        logits = model(input_tensor)

    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    class_map = np.argmax(probs, axis=-1)
    confidence_map = np.max(probs, axis=-1)

    # Create color overlay
    color_overlay = CLASS_COLORS[class_map]

    # Find connected regions for each class
    regions = []
    region_id = 0
    for cls_idx in range(len(CLASS_NAMES)):
        cls_mask = (class_map == cls_idx).astype(np.uint8) * 255
        if cls_mask.sum() == 0:
            continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 100:
                continue

            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            cx, cy = int(centroids[i][0]), int(centroids[i][1])

            regions.append({
                "id": region_id,
                "class": CLASS_NAMES[cls_idx],
                "class_idx": cls_idx,
                "confidence": round(float(np.mean(confidence_map[labels == i])), 2),
                "area": int(area),
                "contour": approx,
                "centroid": (cx, cy),
                "status": "pending"
            })
            region_id += 1

    regions.sort(key=lambda r: r["area"], reverse=True)

    return {
        "class_map": class_map,
        "confidence_map": confidence_map,
        "color_overlay": color_overlay,
        "regions": regions
    }


def render_overlay(image: np.ndarray, result: dict, highlight_region: int = None) -> np.ndarray:
    """Render classification overlay on image."""
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    blended = cv2.addWeighted(image, 0.5, result["color_overlay"], 0.5, 0)

    for region in result["regions"]:
        color = CLASS_COLORS[region["class_idx"]].tolist()
        thickness = 3 if region["id"] == highlight_region else 1
        cv2.drawContours(blended, [region["contour"]], -1, color, thickness)

        if region["id"] == highlight_region:
            cv2.circle(blended, region["centroid"], 5, (255, 255, 255), -1)
            cv2.circle(blended, region["centroid"], 5, color, 2)

    return blended


def process_image(image: np.ndarray) -> dict:
    """Full pipeline: classify and find regions."""
    return classify_image(image)
