import numpy as np
from dw_processor import CLASS_NAMES, CLASS_COLORS, classify_image, extract_regions


def ensemble_predict(dw_probs: np.ndarray, prithvi_probs: np.ndarray,
                     weights: tuple = (0.5, 0.5)) -> dict:
    """Combine DW and Prithvi predictions via weighted average.

    Args:
        dw_probs: (H, W, 9) Dynamic World probabilities
        prithvi_probs: (H, W, 9) Prithvi probabilities
        weights: (dw_weight, prithvi_weight)

    Returns:
        dict with class_map, confidence_map, regions
    """
    w_dw, w_prithvi = weights
    total = w_dw + w_prithvi

    combined_probs = (w_dw * dw_probs + w_prithvi * prithvi_probs) / total

    class_map = np.argmax(combined_probs, axis=-1)
    confidence_map = np.max(combined_probs, axis=-1)

    return {
        "probs": combined_probs,
        "class_map": class_map,
        "confidence_map": confidence_map
    }


def ensemble_classify(s2_array: np.ndarray, confidence_threshold: float = 0.0,
                      dw_weight: float = 0.5, prithvi_weight: float = 0.5) -> dict:
    """Run both models and combine predictions.

    Args:
        s2_array: (H, W, 9) Sentinel-2 array
        confidence_threshold: minimum confidence to keep prediction
        dw_weight: weight for Dynamic World
        prithvi_weight: weight for Prithvi

    Returns:
        dict with class_map, confidence_map, color_overlay, regions
    """
    from prithvi_processor import run_prithvi
    from dw_processor import classify_image as _classify_image

    prithvi_probs = run_prithvi(s2_array)

    dw_result = _classify_image(s2_array, is_sentinel2=True, confidence_threshold=0.0)
    dw_probs = dw_result["probs"]

    result = ensemble_predict(dw_probs, prithvi_probs, (dw_weight, prithvi_weight))

    class_map = result["class_map"]
    confidence_map = result["confidence_map"]

    if confidence_threshold > 0:
        low_conf = confidence_map < confidence_threshold
        class_map[low_conf] = -1

    color_overlay = CLASS_COLORS[class_map.clip(0)]

    regions = extract_regions(class_map, confidence_map)

    return {
        "class_map": class_map,
        "confidence_map": confidence_map,
        "color_overlay": color_overlay,
        "regions": regions,
        "dw_probs": dw_probs,
        "prithvi_probs": prithvi_probs,
        "combined_probs": result["probs"]
    }
