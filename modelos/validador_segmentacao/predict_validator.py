import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    from .utils import (
        IMG_SIZE,
        estimate_non_satellite_score,
        load_classes_json,
        load_image_pil,
        prepare_image_for_model,
    )
except ImportError:
    from utils import (
        IMG_SIZE,
        estimate_non_satellite_score,
        load_classes_json,
        load_image_pil,
        prepare_image_for_model,
    )

MODEL_DIR = Path("modelos/validador_segmentacao")
MODEL_PATH = MODEL_DIR / "validador_segmentacao.keras"
CLASSES_PATH = MODEL_DIR / "classes.json"
MIN_CONFIDENCE_FOR_AUTO_DECISION = 0.60
MIN_MARGIN_FOR_AUTO_DECISION = 0.12

_MODEL = None
_CLASS_MAPPING = None


def _load_artifacts() -> tuple[tf.keras.Model, Dict[int, str]]:
    global _MODEL, _CLASS_MAPPING
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
    if _CLASS_MAPPING is None:
        if not CLASSES_PATH.exists():
            raise FileNotFoundError(f"Arquivo de classes não encontrado: {CLASSES_PATH}")
        _CLASS_MAPPING = load_classes_json(CLASSES_PATH)
    return _MODEL, _CLASS_MAPPING


def _apply_non_satellite_bias(
    probabilities: np.ndarray,
    class_mapping: Dict[int, str],
    image,
) -> tuple[np.ndarray, Dict[str, Any]]:
    heuristic = estimate_non_satellite_score(image)
    adjusted = probabilities.astype(np.float32).copy()

    reject_idx = next((idx for idx, name in class_mapping.items() if name == "rejeitar"), None)
    if reject_idx is None:
        return adjusted, {"aplicada": False, **heuristic}

    score = float(heuristic["score"])
    if score < 0.55:
        return adjusted, {"aplicada": False, **heuristic}

    boost = min(0.55, 0.25 + 0.45 * score)
    adjusted *= (1.0 - boost)
    adjusted[reject_idx] += boost
    adjusted /= adjusted.sum()

    return adjusted, {
        "aplicada": True,
        "score": score,
        "white_ratio": heuristic["white_ratio"],
        "border_mean": heuristic["border_mean"],
        "border_std": heuristic["border_std"],
        "center_std": heuristic["center_std"],
        "edge_density": heuristic["edge_density"],
        "is_likely_non_satellite": heuristic["is_likely_non_satellite"],
    }


def _format_prediction(
    probabilities: np.ndarray,
    class_mapping: Dict[int, str],
    heuristic_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_mapping[predicted_index]
    confidence = float(probabilities[predicted_index])
    ranked_indices = list(np.argsort(probabilities)[::-1])
    second_index = ranked_indices[1] if len(ranked_indices) > 1 else predicted_index
    margin = float(probabilities[predicted_index] - probabilities[second_index])

    probability_map = {
        class_mapping[idx]: float(probabilities[idx])
        for idx in range(len(probabilities))
    }

    final_class = predicted_class
    calibration_reason = None
    if predicted_class != "revisar" and (
        confidence < MIN_CONFIDENCE_FOR_AUTO_DECISION
        or margin < MIN_MARGIN_FOR_AUTO_DECISION
    ):
        final_class = "revisar"
        calibration_reason = (
            "baixa confianca"
            if confidence < MIN_CONFIDENCE_FOR_AUTO_DECISION
            else "classes muito proximas"
        )

    result = {
        "classe": final_class,
        "confianca": confidence,
        "margem": margin,
        "classe_bruta_modelo": predicted_class,
        "probabilidades": probability_map,
        "decisao_calibrada": final_class != predicted_class,
        "limiar_confianca": MIN_CONFIDENCE_FOR_AUTO_DECISION,
        "limiar_margem": MIN_MARGIN_FOR_AUTO_DECISION,
    }
    if calibration_reason is not None:
        result["motivo_calibracao"] = calibration_reason
    if heuristic_info is not None:
        result["heuristica_fora_do_dominio"] = heuristic_info
    return result


def prever_qualidade_segmentacao(caminho_imagem: str | Path) -> Dict[str, Any]:
    model, class_mapping = _load_artifacts()
    image = load_image_pil(caminho_imagem)
    image_array = prepare_image_for_model(image, img_size=IMG_SIZE)
    image_array = preprocess_input(image_array)
    probabilities = model.predict(image_array, verbose=0)[0]
    probabilities, heuristic_info = _apply_non_satellite_bias(probabilities, class_mapping, image)
    return _format_prediction(probabilities, class_mapping, heuristic_info)


def prever_qualidade_segmentacao_pil(imagem_pil) -> Dict[str, Any]:
    model, class_mapping = _load_artifacts()
    image_array = prepare_image_for_model(imagem_pil, img_size=IMG_SIZE)
    image_array = preprocess_input(image_array)
    probabilities = model.predict(image_array, verbose=0)[0]
    probabilities, heuristic_info = _apply_non_satellite_bias(probabilities, class_mapping, imagem_pil)
    return _format_prediction(probabilities, class_mapping, heuristic_info)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediz a qualidade da segmentação a partir de uma imagem lado a lado.")
    parser.add_argument("--image", required=True, help="Caminho da imagem lado a lado")
    args = parser.parse_args()

    resultado = prever_qualidade_segmentacao(args.image)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
