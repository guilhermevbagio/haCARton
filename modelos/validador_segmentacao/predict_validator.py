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
    from .utils import IMG_SIZE, load_classes_json, load_image_pil, prepare_image_for_model
except ImportError:
    from utils import IMG_SIZE, load_classes_json, load_image_pil, prepare_image_for_model

MODEL_DIR = Path("modelos/validador_segmentacao")
MODEL_PATH = MODEL_DIR / "validador_segmentacao.keras"
CLASSES_PATH = MODEL_DIR / "classes.json"

_MODEL = None
_CLASS_MAPPING = None


def _load_artifacts() -> tuple[tf.keras.Model, Dict[int, str]]:
    global _MODEL, _CLASS_MAPPING
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo n?o encontrado: {MODEL_PATH}")
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
    if _CLASS_MAPPING is None:
        if not CLASSES_PATH.exists():
            raise FileNotFoundError(f"Arquivo de classes n?o encontrado: {CLASSES_PATH}")
        _CLASS_MAPPING = load_classes_json(CLASSES_PATH)
    return _MODEL, _CLASS_MAPPING


def _format_prediction(probabilities: np.ndarray, class_mapping: Dict[int, str]) -> Dict[str, Any]:
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_mapping[predicted_index]
    confidence = float(probabilities[predicted_index])
    probability_map = {
        class_mapping[idx]: float(probabilities[idx])
        for idx in range(len(probabilities))
    }
    return {
        "classe": predicted_class,
        "confianca": confidence,
        "probabilidades": probability_map,
    }


def prever_qualidade_segmentacao(caminho_imagem: str | Path) -> Dict[str, Any]:
    model, class_mapping = _load_artifacts()
    image = load_image_pil(caminho_imagem)
    image_array = prepare_image_for_model(image, img_size=IMG_SIZE)
    image_array = preprocess_input(image_array)
    probabilities = model.predict(image_array, verbose=0)[0]
    return _format_prediction(probabilities, class_mapping)


def prever_qualidade_segmentacao_pil(imagem_pil) -> Dict[str, Any]:
    model, class_mapping = _load_artifacts()
    image_array = prepare_image_for_model(imagem_pil, img_size=IMG_SIZE)
    image_array = preprocess_input(image_array)
    probabilities = model.predict(image_array, verbose=0)[0]
    return _format_prediction(probabilities, class_mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediz a qualidade da segmenta??o a partir de uma imagem lado a lado.")
    parser.add_argument("--image", required=True, help="Caminho da imagem lado a lado")
    args = parser.parse_args()

    resultado = prever_qualidade_segmentacao(args.image)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
