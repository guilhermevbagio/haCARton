import colorsys
from typing import Dict, List, Tuple

import cv2
import numpy as np

CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Corpos d'água": (40, 120, 255),
    "Vegetação nativa/candidata": (46, 160, 67),
    "Área antropizada/consolidada": (255, 180, 45),
    "Indefinido / baixa confiança": (153, 102, 204),
}

SEGMENTER_LABELS = {
    "sam2": "SAM2 padrão (protótipo heurístico)",
    "rsam_seg": "RSAM-Seg experimental (fallback para heurística)",
}


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image[:, :, :3].copy()


def generate_segmentation_masks(
    image: np.ndarray,
    max_regions: int = 18,
    min_area_ratio: float = 0.0025,
) -> List[np.ndarray]:
    """Approximate region proposals using image-driven clustering.

    This keeps the prototype deterministic and visually coherent while the
    production segmenter (SAM2/RSAM-Seg) is not wired in yet.
    """
    rgb = ensure_rgb(image)
    height, width = rgb.shape[:2]
    total_pixels = height * width
    min_area = max(64, int(total_pixels * min_area_ratio))

    blurred = cv2.GaussianBlur(rgb, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    samples = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.5,
    )
    _compactness, labels, _centers = cv2.kmeans(
        samples,
        4,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    label_map = labels.reshape((height, width))
    masks: List[np.ndarray] = []
    kernel = np.ones((3, 3), np.uint8)

    for cluster_id in range(4):
        cluster_mask = np.where(label_map == cluster_id, 255, 0).astype(np.uint8)
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_OPEN, kernel)
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel)

        num_components, components = cv2.connectedComponents(cluster_mask)
        for component_id in range(1, num_components):
            component_mask = np.where(components == component_id, 255, 0).astype(np.uint8)
            area = int(np.count_nonzero(component_mask))
            if area < min_area:
                continue
            masks.append(component_mask)

    masks.sort(key=lambda mask: int(np.count_nonzero(mask)), reverse=True)
    if not masks:
        fallback_mask = np.full((height, width), 255, dtype=np.uint8)
        masks.append(fallback_mask)
    return masks[:max_regions]


def extract_mask_features(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    rgb = ensure_rgb(image)
    region = mask > 0
    pixels = rgb[region]
    if pixels.size == 0:
        return {
            "area_pixels": 0,
            "mean_r": 0.0,
            "mean_g": 0.0,
            "mean_b": 0.0,
            "brightness": 0.0,
            "saturation": 0.0,
            "green_dominance": 0.0,
            "blue_dominance": 0.0,
            "texture": 0.0,
        }

    mean_rgb = pixels.mean(axis=0) / 255.0
    hsv_pixels = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in pixels / 255.0])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_pixels = gray[region] / 255.0

    return {
        "area_pixels": int(region.sum()),
        "mean_r": float(mean_rgb[0]),
        "mean_g": float(mean_rgb[1]),
        "mean_b": float(mean_rgb[2]),
        "brightness": float(hsv_pixels[:, 2].mean()),
        "saturation": float(hsv_pixels[:, 1].mean()),
        "green_dominance": float(mean_rgb[1] - max(mean_rgb[0], mean_rgb[2])),
        "blue_dominance": float(mean_rgb[2] - max(mean_rgb[0], mean_rgb[1])),
        "texture": float(gray_pixels.std()),
    }


def classificar_mascara_rgb(image: np.ndarray, mask: np.ndarray) -> Dict[str, object]:
    features = extract_mask_features(image, mask)
    area = features["area_pixels"]

    if area == 0:
        return {
            "class": "Indefinido / baixa confiança",
            "confidence": 0.0,
            "features": features,
            "review_reason": "Máscara vazia",
        }

    blue_score = (
        1.6 * max(0.0, features["blue_dominance"])
        + 0.6 * features["saturation"]
        + 0.3 * features["brightness"]
        - 0.4 * features["texture"]
    )
    vegetation_score = (
        1.8 * max(0.0, features["green_dominance"])
        + 0.7 * features["saturation"]
        + 0.4 * features["texture"]
        + 0.2 * features["brightness"]
    )
    anthropized_score = (
        0.9 * features["brightness"]
        + 0.5 * (1.0 - features["saturation"])
        + 0.4 * max(features["mean_r"], features["mean_g"])
        + 0.3 * features["texture"]
    )

    scores = {
        "Corpos d'água": blue_score,
        "Vegetação nativa/candidata": vegetation_score,
        "Área antropizada/consolidada": anthropized_score,
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ordered[0]
    second_score = ordered[1][1]
    margin = max(0.0, best_score - second_score)
    confidence = float(np.clip(0.45 + margin, 0.0, 0.98))
    review_reason = ""

    if best_label == "Corpos d'água" and features["blue_dominance"] < 0.04:
        confidence *= 0.75
        review_reason = "Assinatura azul fraca para água"
    elif best_label == "Vegetação nativa/candidata" and features["green_dominance"] < 0.03:
        confidence *= 0.75
        review_reason = "Predominância de verde pouco marcada"
    elif best_label == "Área antropizada/consolidada" and features["brightness"] < 0.35:
        confidence *= 0.8
        review_reason = "Brilho baixo para área antropizada"

    if confidence < 0.6:
        best_label = "Indefinido / baixa confiança"
        review_reason = review_reason or "Separação heurística inconclusiva"

    return {
        "class": best_label,
        "confidence": round(float(confidence), 2),
        "features": features,
        "review_reason": review_reason,
    }


def classificar_mascaras(image: np.ndarray, masks: List[np.ndarray]) -> List[Dict[str, object]]:
    classifications: List[Dict[str, object]] = []
    for idx, mask in enumerate(masks, start=1):
        classification = classificar_mascara_rgb(image, mask)
        classifications.append(
            {
                "id": idx,
                "mask": mask,
                "class": classification["class"],
                "confidence": classification["confidence"],
                "status": "pending",
                "area_pixels": classification["features"]["area_pixels"],
                "features": classification["features"],
                "review_reason": classification["review_reason"],
                "needs_review": classification["confidence"] < 0.7
                or classification["class"] == "Indefinido / baixa confiança",
            }
        )
    return classifications


def _blend_overlay(base: np.ndarray, overlay: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    return cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)


def build_segmentation_overlay(image: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    rgb = ensure_rgb(image)
    overlay = rgb.copy()
    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (255, 215, 0),
        (186, 85, 211),
        (0, 206, 209),
    ]

    for idx, mask in enumerate(masks):
        color = palette[idx % len(palette)]
        overlay[mask > 0] = color
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return _blend_overlay(rgb, overlay)


def montar_overlay_classificado(image: np.ndarray, classifications: List[Dict[str, object]]) -> np.ndarray:
    rgb = ensure_rgb(image)
    overlay = rgb.copy()

    for item in classifications:
        color = CLASS_COLORS[item["class"]]
        mask = item["mask"]
        overlay[mask > 0] = color
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return _blend_overlay(rgb, overlay)


def gerar_resumo_classificacao(classifications: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for class_name in CLASS_COLORS:
        class_items = [item for item in classifications if item["class"] == class_name]
        total_area = sum(int(item["area_pixels"]) for item in class_items)
        avg_confidence = (
            sum(float(item["confidence"]) for item in class_items) / len(class_items)
            if class_items
            else 0.0
        )
        summary.append(
            {
                "Classe": class_name,
                "Regiões": len(class_items),
                "Área aproximada (px)": total_area,
                "Confiança média": round(avg_confidence, 2),
            }
        )
    return summary


def evaluate_rsam_seg_support() -> Dict[str, object]:
    """Static integration note based on repository review.

    The codebase is prepared so a real RSAM-Seg adapter can replace the
    heuristic pipeline once checkpoints and an inference workflow are available
    in the target environment.
    """
    return {
        "available_now": False,
        "segmenter_used": SEGMENTER_LABELS["sam2"],
        "requested_segmenter": SEGMENTER_LABELS["rsam_seg"],
        "message": (
            "RSAM-Seg permanece como trilha experimental. O protótipo atual faz "
            "fallback para a segmentação heurística porque a integração imediata "
            "depende de pesos, configuração e rotina de inferência dedicados."
        ),
    }


def process_image(image: np.ndarray, segmenter: str = "sam2") -> Dict[str, object]:
    rgb = ensure_rgb(image)
    masks = generate_segmentation_masks(rgb)
    classifications = classificar_mascaras(rgb, masks)
    summary = gerar_resumo_classificacao(classifications)

    rsam_note = None
    segmenter_used = SEGMENTER_LABELS["sam2"]
    if segmenter == "rsam_seg":
        rsam_note = evaluate_rsam_seg_support()
        segmenter_used = rsam_note["segmenter_used"]

    return {
        "segmenter_requested": SEGMENTER_LABELS.get(segmenter, SEGMENTER_LABELS["sam2"]),
        "segmenter_used": segmenter_used,
        "rsam_note": rsam_note,
        "regions": classifications,
        "segmentation_overlay": build_segmentation_overlay(rgb, masks),
        "classification_overlay": montar_overlay_classificado(rgb, classifications),
        "summary": summary,
        "prototype_message": (
            "Este protótipo demonstra a pré-classificação automática de regiões "
            "segmentadas em imagem de satélite. A classificação atual é "
            "demonstrativa. Em produção, a etapa de classificação seria integrada "
            "a bases/modelos como Dynamic World, MapBiomas, SNIF e/ou modelos "
            "especializados em sensoriamento remoto, com cruzamento temporal para "
            "separar área consolidada de área antropizada após 22/07/2008. "
            "Regiões com baixa confiança seriam enviadas para revisão humana."
        ),
    }
