from __future__ import annotations

import importlib.util
from typing import Dict

import numpy as np

from mock_processor import process_image as mock_process_image


SEGMENTER_RUNTIME_LABELS = {
    "sam2": "SAM2 padrão (protótipo heurístico)",
    "samgeo": "SAMGeo experimental",
    "rsam_seg": "RSAM-Seg experimental",
}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def get_segmenter_availability() -> Dict[str, Dict[str, object]]:
    return {
        "sam2": {
            "available": True,
            "message": "Backend heurístico local disponível.",
        },
        "samgeo": {
            "available": _module_available("samgeo"),
            "message": (
                "SAMGeo não está instalado no ambiente atual. Instale a dependência para testar esse backend."
                if not _module_available("samgeo")
                else "SAMGeo encontrado no ambiente; integração pronta para teste experimental."
            ),
        },
        "rsam_seg": {
            "available": False,
            "message": "RSAM-Seg ainda depende de pacote/adaptador/checkpoint específico neste projeto.",
        },
    }


def _append_backend_note(result: Dict[str, object], message: str, segmenter_key: str) -> Dict[str, object]:
    result = dict(result)
    result["backend_note"] = {
        "segmenter_key": segmenter_key,
        "message": message,
    }
    return result


def _run_samgeo(image: np.ndarray) -> Dict[str, object]:
    availability = get_segmenter_availability()["samgeo"]
    if not availability["available"]:
        raise RuntimeError(str(availability["message"]))

    # Experimental adapter: while SAMGeo-specific prompt/checkpoint handling is not
    # standardized in this prototype, we reuse the current visualization pipeline and
    # clearly label the output so the Streamlit UI can surface the backend choice.
    result = mock_process_image(image, segmenter="sam2")
    result["segmenter_requested"] = SEGMENTER_RUNTIME_LABELS["samgeo"]
    result["segmenter_used"] = SEGMENTER_RUNTIME_LABELS["samgeo"]
    return _append_backend_note(
        result,
        "SAMGeo está em modo experimental nesta demo. O fluxo visual usa o pipeline atual até conectarmos a inferência geoespacial nativa.",
        "samgeo",
    )


def run_segmentation_backend(image: np.ndarray, segmenter_key: str) -> Dict[str, object]:
    if segmenter_key in {"sam2", "rsam_seg"}:
        return mock_process_image(image, segmenter=segmenter_key)
    if segmenter_key == "samgeo":
        return _run_samgeo(image)
    raise ValueError(f"Segmentador não suportado: {segmenter_key}")
