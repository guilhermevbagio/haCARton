from datetime import datetime
from pathlib import Path
import uuid

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from mock_processor import CLASS_COLORS, process_image

PROTOTYPE_DISCLAIMER = (
    "Prot?tipo demonstrativo: a separa??o oficial entre ?rea consolidada e "
    "?rea antropizada ap?s 22/07/2008 depende de cruzamento temporal com bases "
    "como MapBiomas/SNIF e n?o pode ser conclu?da apenas pela imagem atual."
)

SEGMENTER_OPTIONS = {
    "SAM2 padr?o (prot?tipo heur?stico)": "sam2",
    "RSAM-Seg experimental": "rsam_seg",
}

DATASET_CLASSES = {
    "aprovar": "Aprovar",
    "rejeitar": "Rejeitar",
    "revisar": "Revisar",
}

DEEPGLOBE_IGNORE_TERMS = [
    "mask",
    "label",
    "class",
    "gt",
    "groundtruth",
    "ground_truth",
]
DEEPGLOBE_MAX_DIM = 1024
VALIDATOR_MODEL_PATH = Path("modelos/validador_segmentacao/validador_segmentacao.keras")
VALIDATOR_CLASSES_PATH = Path("modelos/validador_segmentacao/classes.json")


def apply_global_styles() -> None:
    st.markdown(
        """
<style>
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) > div > button {
        background-color: #28a745 !important;
        color: white !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) > div > button {
        background-color: #dc3545 !important;
        color: white !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(4) > div > button {
        background-color: #ffc107 !important;
        color: black !important;
    }
    .stAlert {
        border-radius: 12px;
    }
    .stImage img {
        max-height: 70vh;
        object-fit: contain;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def converter_para_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img.astype(np.uint8), mode="L").convert("RGB")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.shape[2] == 4:
            return Image.fromarray(img, mode="RGBA").convert("RGB")
        return Image.fromarray(img).convert("RGB")

    raise TypeError(f"Tipo de imagem n?o suportado: {type(img)}")


def montar_imagem_lado_a_lado(imagem_original, imagem_segmentada):
    original = converter_para_pil(imagem_original)
    segmentada = converter_para_pil(imagem_segmentada)
    segmentada = segmentada.resize(original.size)

    largura, altura = original.size
    final = Image.new("RGB", (largura * 2, altura))
    final.paste(original, (0, 0))
    final.paste(segmentada, (largura, 0))
    return final


def salvar_lado_a_lado(imagem_original, imagem_segmentada, classe_destino, base_dir="dataset_rotulado"):
    classes_validas = list(DATASET_CLASSES.keys())
    if classe_destino not in classes_validas:
        raise ValueError(f"classe_destino deve ser uma de: {classes_validas}")

    final = montar_imagem_lado_a_lado(imagem_original, imagem_segmentada)

    base_path = Path(base_dir)
    for folder_name in classes_validas:
        (base_path / folder_name).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    output_path = base_path / classe_destino / f"ex_{timestamp}_{uid}.png"
    final.save(output_path)
    return str(output_path)


def encontrar_imagens_dataset(dataset_path, max_images=100):
    dataset_path = Path(dataset_path)
    extensoes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    imagens = []

    for arquivo in dataset_path.rglob("*"):
        if arquivo.suffix.lower() not in extensoes:
            continue
        nome_lower = arquivo.name.lower()
        if any(palavra in nome_lower for palavra in DEEPGLOBE_IGNORE_TERMS):
            continue
        imagens.append(str(arquivo))

    imagens = sorted(imagens)
    return imagens[:max_images]


def carregar_dataset_local(dataset_path, max_images=100):
    dataset_path = Path(dataset_path).expanduser()
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Pasta do dataset n?o encontrada: {dataset_path}")

    imagens = encontrar_imagens_dataset(dataset_path, max_images=max_images)
    return str(dataset_path), imagens


def preparar_imagem_para_revisao(image_np, max_dim=DEEPGLOBE_MAX_DIM):
    if image_np.ndim == 2:
        height, width = image_np.shape
    else:
        height, width = image_np.shape[:2]

    largest_dim = max(height, width)
    if largest_dim <= max_dim:
        return image_np

    scale = max_dim / float(largest_dim)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image_np, (new_width, new_height), interpolation=interpolation)


def ensure_session_defaults():
    defaults = {
        "analysis": None,
        "current_idx": 0,
        "uploaded_image": None,
        "uploaded_filename": None,
        "imagem_original": None,
        "imagem_segmentada": None,
        "rotulo_dataset_path": None,
        "deepglobe_imagens": [],
        "deepglobe_idx": 0,
        "deepglobe_total": 0,
        "deepglobe_counts": {"aprovar": 0, "rejeitar": 0, "revisar": 0},
        "deepglobe_dataset_path": None,
        "deepglobe_current_path": None,
        "deepglobe_current_original": None,
        "deepglobe_current_segmented": None,
        "deepglobe_current_analysis": None,
        "deepglobe_current_error": None,
        "deepglobe_last_saved_path": None,
        "deepglobe_input_path": "",
        "validator_manual_key": None,
        "validator_manual_result": None,
        "validator_manual_error": None,
        "validator_deepglobe_key": None,
        "validator_deepglobe_result": None,
        "validator_deepglobe_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def processar_imagem_com_segmentador(image_np, segmenter_key):
    return process_image(image_np, segmenter=segmenter_key)


def validador_disponivel():
    return VALIDATOR_MODEL_PATH.exists() and VALIDATOR_CLASSES_PATH.exists()


def prever_validacao_automatica(imagem_original, imagem_segmentada):
    from modelos.validador_segmentacao.predict_validator import prever_qualidade_segmentacao_pil

    imagem_lado_a_lado = montar_imagem_lado_a_lado(imagem_original, imagem_segmentada)
    return prever_qualidade_segmentacao_pil(imagem_lado_a_lado)


def obter_validacao_manual(upload_key, imagem_original, imagem_segmentada):
    if not validador_disponivel():
        return None, "Modelo de valida??o autom?tica ainda n?o foi treinado/carregado."

    if st.session_state.validator_manual_key == upload_key:
        return st.session_state.validator_manual_result, st.session_state.validator_manual_error

    try:
        resultado = prever_validacao_automatica(imagem_original, imagem_segmentada)
        st.session_state.validator_manual_key = upload_key
        st.session_state.validator_manual_result = resultado
        st.session_state.validator_manual_error = None
        return resultado, None
    except Exception as exc:
        st.session_state.validator_manual_key = upload_key
        st.session_state.validator_manual_result = None
        st.session_state.validator_manual_error = str(exc)
        return None, str(exc)


def obter_validacao_deepglobe(image_key, imagem_original, imagem_segmentada):
    if not validador_disponivel():
        return None, "Modelo de valida??o autom?tica ainda n?o foi treinado/carregado."

    if st.session_state.validator_deepglobe_key == image_key:
        return st.session_state.validator_deepglobe_result, st.session_state.validator_deepglobe_error

    try:
        resultado = prever_validacao_automatica(imagem_original, imagem_segmentada)
        st.session_state.validator_deepglobe_key = image_key
        st.session_state.validator_deepglobe_result = resultado
        st.session_state.validator_deepglobe_error = None
        return resultado, None
    except Exception as exc:
        st.session_state.validator_deepglobe_key = image_key
        st.session_state.validator_deepglobe_result = None
        st.session_state.validator_deepglobe_error = str(exc)
        return None, str(exc)


def render_validator_result(resultado, titulo="Valida??o autom?tica da segmenta??o"):
    st.subheader(titulo)
    classe = resultado["classe"]
    confianca = float(resultado["confianca"])
    probabilidades = resultado["probabilidades"]

    if classe == "aprovar":
        st.success(f"Recomenda??o do modelo: {classe} ({confianca:.0%})")
    elif classe == "rejeitar":
        st.error(f"Recomenda??o do modelo: {classe} ({confianca:.0%})")
    else:
        st.warning(f"Recomenda??o do modelo: {classe} ({confianca:.0%})")

    prob_rows = [
        {"Classe": nome, "Probabilidade": round(float(valor), 4)}
        for nome, valor in probabilidades.items()
    ]
    st.dataframe(prob_rows, use_container_width=True, hide_index=True)


def carregar_analise_deepglobe(segmenter_key):
    imagens = st.session_state.deepglobe_imagens
    idx = st.session_state.deepglobe_idx
    if not imagens or idx >= len(imagens):
        return

    current_path = imagens[idx]
    if st.session_state.deepglobe_current_path == current_path and (
        st.session_state.deepglobe_current_analysis is not None or st.session_state.deepglobe_current_error
    ):
        return

    try:
        image = Image.open(current_path)
        image_np = np.array(image)
        image_np = preparar_imagem_para_revisao(image_np)
        analysis = processar_imagem_com_segmentador(image_np, segmenter_key)
        st.session_state.deepglobe_current_path = current_path
        st.session_state.deepglobe_current_original = image_np
        st.session_state.deepglobe_current_segmented = analysis["segmentation_overlay"]
        st.session_state.deepglobe_current_analysis = analysis
        st.session_state.deepglobe_current_error = None
    except Exception as exc:
        st.session_state.deepglobe_current_path = current_path
        st.session_state.deepglobe_current_original = None
        st.session_state.deepglobe_current_segmented = None
        st.session_state.deepglobe_current_analysis = None
        st.session_state.deepglobe_current_error = str(exc)


def avancar_deepglobe():
    st.session_state.deepglobe_idx += 1
    st.session_state.deepglobe_current_path = None
    st.session_state.deepglobe_current_original = None
    st.session_state.deepglobe_current_segmented = None
    st.session_state.deepglobe_current_analysis = None
    st.session_state.deepglobe_current_error = None
    st.session_state.validator_deepglobe_key = None
    st.session_state.validator_deepglobe_result = None
    st.session_state.validator_deepglobe_error = None


def rotular_deepglobe(classe_destino):
    imagem_original = st.session_state.get("deepglobe_current_original")
    imagem_segmentada = st.session_state.get("deepglobe_current_segmented")
    if imagem_original is None or imagem_segmentada is None:
        st.error("Gere a segmenta??o antes de rotular este exemplo.")
        return

    caminho = salvar_lado_a_lado(imagem_original, imagem_segmentada, classe_destino)
    st.session_state.deepglobe_counts[classe_destino] += 1
    st.session_state.deepglobe_last_saved_path = caminho
    st.session_state.rotulo_dataset_path = caminho
    avancar_deepglobe()
    st.rerun()


def reiniciar_revisao_deepglobe():
    st.session_state.deepglobe_idx = 0
    st.session_state.deepglobe_counts = {"aprovar": 0, "rejeitar": 0, "revisar": 0}
    st.session_state.deepglobe_current_path = None
    st.session_state.deepglobe_current_original = None
    st.session_state.deepglobe_current_segmented = None
    st.session_state.deepglobe_current_analysis = None
    st.session_state.deepglobe_current_error = None
    st.session_state.deepglobe_last_saved_path = None
    st.session_state.rotulo_dataset_path = None
    st.session_state.validator_deepglobe_key = None
    st.session_state.validator_deepglobe_result = None
    st.session_state.validator_deepglobe_error = None


def render_manual_label_buttons():
    st.subheader("Rotular qualidade da segmenta??o")
    st.caption(
        "Salva uma imagem lado a lado com a original ? esquerda e o overlay da segmenta??o ? direita."
    )

    label_col1, label_col2, label_col3 = st.columns(3)
    for col, classe_destino in zip(
        [label_col1, label_col2, label_col3],
        ["aprovar", "rejeitar", "revisar"],
    ):
        with col:
            if st.button(DATASET_CLASSES[classe_destino], key=f"dataset_{classe_destino}"):
                imagem_original = st.session_state.get("imagem_original")
                imagem_segmentada = st.session_state.get("imagem_segmentada")
                if imagem_original is None or imagem_segmentada is None:
                    st.error("Gere a segmenta??o antes de rotular este exemplo.")
                else:
                    caminho = salvar_lado_a_lado(
                        imagem_original,
                        imagem_segmentada,
                        classe_destino,
                    )
                    st.session_state.rotulo_dataset_path = caminho
                    st.success(f"Exemplo salvo em: {caminho}")


def render_deepglobe_mode(segmenter_key):
    st.subheader("Modo Coleta com DeepGlobe")
    st.caption(
        "Usa uma pasta local do DeepGlobe j? baixada, passa no fluxo atual de segmenta??o e permite rotular em lote."
    )

    dataset_path_input = st.text_input(
        "Pasta local do DeepGlobe",
        value=st.session_state.deepglobe_input_path,
        placeholder=r"Ex.: C:\Users\Miguel Lucas\Downloads\deepglobe-land-cover-classification-dataset",
    )
    st.session_state.deepglobe_input_path = dataset_path_input

    header1, header2 = st.columns([1, 1])
    with header1:
        if st.button("Carregar imagens da pasta local"):
            try:
                with st.spinner("Lendo imagens do dataset local..."):
                    dataset_path, imagens = carregar_dataset_local(dataset_path_input, max_images=100)
                st.session_state.validator_deepglobe_key = None
                st.session_state.validator_deepglobe_result = None
                st.session_state.validator_deepglobe_error = None
                st.session_state.deepglobe_imagens = imagens
                st.session_state.deepglobe_dataset_path = dataset_path
                st.session_state.deepglobe_total = len(imagens)
                reiniciar_revisao_deepglobe()
                if imagens:
                    st.success(f"Dataset local carregado com {len(imagens)} imagens eleg?veis.")
                else:
                    st.warning("Nenhuma imagem RGB eleg?vel foi encontrada nessa pasta.")
                st.rerun()
            except Exception as exc:
                st.error(f"Falha ao carregar imagens locais: {exc}")

    with header2:
        if st.button("Reiniciar revis?o"):
            if st.session_state.deepglobe_imagens:
                reiniciar_revisao_deepglobe()
                st.rerun()

    imagens = st.session_state.deepglobe_imagens
    total = st.session_state.deepglobe_total
    idx = st.session_state.deepglobe_idx

    if st.session_state.deepglobe_dataset_path:
        st.caption(f"Dataset local: {st.session_state.deepglobe_dataset_path}")

    if not imagens:
        st.info("Informe a pasta local do DeepGlobe e carregue as imagens para iniciar a coleta em lote.")
        return

    count1, count2, count3, count4 = st.columns(4)
    count1.metric("Aprovadas", st.session_state.deepglobe_counts["aprovar"])
    count2.metric("Rejeitadas", st.session_state.deepglobe_counts["rejeitar"])
    count3.metric("Revisar", st.session_state.deepglobe_counts["revisar"])
    count4.metric("Total", total)

    if idx >= total:
        st.success("Revis?o em lote conclu?da para as imagens carregadas.")
        if st.session_state.deepglobe_last_saved_path:
            st.info(f"?ltimo exemplo salvo: {st.session_state.deepglobe_last_saved_path}")
        return

    with st.spinner("Processando imagem atual do DeepGlobe..."):
        carregar_analise_deepglobe(segmenter_key)
    current_path = imagens[idx]

    st.progress((idx + 1) / total)
    st.write(f"Imagem {idx + 1} de {total}")
    st.caption(f"Arquivo atual: {current_path}")
    st.caption(
        f"As imagens do lote s?o redimensionadas para no m?ximo {DEEPGLOBE_MAX_DIM}px antes da segmenta??o para evitar travamentos."
    )

    if st.session_state.deepglobe_current_error:
        st.error(f"Falha na segmenta??o desta imagem: {st.session_state.deepglobe_current_error}")
        error_col1, error_col2 = st.columns(2)
        with error_col1:
            if st.button("Pular imagem", key="deepglobe_skip_error"):
                avancar_deepglobe()
                st.rerun()
        with error_col2:
            if st.button("Tentar novamente", key="deepglobe_retry_error"):
                st.session_state.deepglobe_current_path = None
                st.session_state.deepglobe_current_error = None
                st.rerun()
        return

    analysis = st.session_state.deepglobe_current_analysis
    original = st.session_state.deepglobe_current_original
    segmented = st.session_state.deepglobe_current_segmented

    if analysis is None or original is None or segmented is None:
        st.warning("A an?lise desta imagem ainda n?o est? pronta.")
        return

    viz1, viz2 = st.columns(2)
    with viz1:
        st.image(original, caption="Imagem original", use_container_width=True)
    with viz2:
        st.image(segmented, caption="Segmenta??o autom?tica", use_container_width=True)

    validator_result, validator_error = obter_validacao_deepglobe(current_path, original, segmented)
    if validator_result is not None:
        render_validator_result(validator_result, titulo="Valida??o autom?tica do lote")
    elif validator_error and validador_disponivel():
        st.warning(f"N?o foi poss?vel executar o validador autom?tico neste exemplo: {validator_error}")
    else:
        st.info("Treine o validador para receber uma recomenda??o autom?tica neste modo.")

    action1, action2, action3, action4 = st.columns(4)
    with action1:
        if st.button("Aprovar", key="deepglobe_approve"):
            rotular_deepglobe("aprovar")
    with action2:
        if st.button("Rejeitar", key="deepglobe_reject"):
            rotular_deepglobe("rejeitar")
    with action3:
        if st.button("Revisar", key="deepglobe_review"):
            rotular_deepglobe("revisar")
    with action4:
        if st.button("Pular imagem", key="deepglobe_skip"):
            avancar_deepglobe()
            st.rerun()

    if st.session_state.deepglobe_last_saved_path:
        st.info(f"?ltimo exemplo salvo: {st.session_state.deepglobe_last_saved_path}")


def render_manual_upload_mode(segmenter_key):
    uploaded_file = st.file_uploader(
        "Upload da imagem de sat?lite",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        if st.session_state.uploaded_filename != uploaded_file.name:
            st.session_state.analysis = None
            st.session_state.current_idx = 0
            st.session_state.rotulo_dataset_path = None
            st.session_state.imagem_segmentada = None
        st.session_state.uploaded_image = image_np
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.imagem_original = image_np

    image_np = st.session_state.uploaded_image
    if image_np is None:
        return

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.subheader("Imagem original")
        st.image(image_np, use_container_width=True)

    with col2:
        st.subheader("Objetivo da demonstra??o")
        st.info(PROTOTYPE_DISCLAIMER)
        st.markdown(
            """
- Segmenta??o autom?tica de regi?es
- Pr?-classifica??o por classe demonstrativa
- Regi?es de baixa confian?a destacadas para revis?o humana
"""
        )
        if st.button("Processar imagem"):
            with st.spinner("Executando segmenta??o e classifica??o..."):
                st.session_state.analysis = processar_imagem_com_segmentador(
                    image_np,
                    segmenter_key,
                )
                st.session_state.current_idx = 0
                st.session_state.rotulo_dataset_path = None
                st.session_state.imagem_original = image_np
                st.session_state.imagem_segmentada = st.session_state.analysis["segmentation_overlay"]
                st.session_state.validator_manual_key = None
                st.session_state.validator_manual_result = None
                st.session_state.validator_manual_error = None
            st.rerun()

    analysis = st.session_state.analysis
    if not analysis:
        return

    regions = analysis["regions"]
    st.session_state.imagem_original = image_np
    st.session_state.imagem_segmentada = analysis["segmentation_overlay"]

    st.divider()
    st.subheader("Resultado do processamento")

    if analysis["rsam_note"]:
        st.warning(analysis["rsam_note"]["message"])

    meta1, meta2, meta3, meta4 = st.columns(4)
    meta1.metric("Segmentador solicitado", analysis["segmenter_requested"])
    meta2.metric("Segmentador usado", analysis["segmenter_used"])
    meta3.metric("Regi?es detectadas", len(regions))
    meta4.metric(
        "Baixa confian?a",
        sum(1 for item in regions if item["needs_review"]),
    )

    viz1, viz2, viz3 = st.columns(3)
    with viz1:
        st.image(image_np, caption="Imagem original", use_container_width=True)
    with viz2:
        st.image(
            analysis["segmentation_overlay"],
            caption="Segmenta??o autom?tica",
            use_container_width=True,
        )
    with viz3:
        st.image(
            analysis["classification_overlay"],
            caption="Classifica??o colorida por classe",
            use_container_width=True,
        )

    upload_key = f"{st.session_state.uploaded_filename}:{analysis['segmenter_used']}"
    validator_result, validator_error = obter_validacao_manual(
        upload_key,
        image_np,
        analysis["segmentation_overlay"],
    )
    if validator_result is not None:
        render_validator_result(validator_result)
    elif validator_error and validador_disponivel():
        st.warning(f"N?o foi poss?vel executar o validador autom?tico: {validator_error}")
    else:
        st.info("Treine o validador para receber uma recomenda??o autom?tica da qualidade da segmenta??o.")

    render_manual_label_buttons()

    if st.session_state.rotulo_dataset_path:
        st.info(f"?ltimo exemplo salvo: {st.session_state.rotulo_dataset_path}")

    st.caption(analysis["prototype_message"])

    st.subheader("Resumo por classe")
    st.dataframe(analysis["summary"], use_container_width=True, hide_index=True)

    class_count_cols = st.columns(len(CLASS_COLORS))
    for col, class_name in zip(class_count_cols, CLASS_COLORS.keys()):
        count = sum(1 for item in regions if item["class"] == class_name)
        col.metric(class_name, count)

    low_confidence = [
        {
            "Regi?o": item["id"],
            "Classe": item["class"],
            "Confian?a": item["confidence"],
            "Motivo": item["review_reason"] or "Encaminhar para revis?o humana",
        }
        for item in regions
        if item["needs_review"]
    ]

    if low_confidence:
        st.subheader("Regi?es sugeridas para revis?o humana")
        st.dataframe(low_confidence, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Revis?o regi?o por regi?o")

    if not regions:
        st.error("Nenhuma regi?o eleg?vel foi encontrada para revis?o.")
        st.stop()

    idx = min(st.session_state.current_idx, len(regions) - 1)
    result = regions[idx]

    overlay = image_np.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
    elif overlay.shape[2] == 4:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2RGB)

    image_display = overlay.copy()
    mask = result["mask"]
    overlay[mask > 0] = CLASS_COLORS[result["class"]]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    blended = cv2.addWeighted(image_display, 0.58, overlay, 0.42, 0)

    st.image(blended, caption=f"Regi?o {result['id']}", use_container_width=True)

    info1, info2, info3, info4 = st.columns(4)
    info1.write(f"**Classe:** {result['class']}")
    info2.write(f"**Confian?a:** {result['confidence']:.0%}")
    info3.write(f"**?rea (px):** {result['area_pixels']}")
    info4.write(f"**Status:** {result['status']}")

    if result["review_reason"]:
        st.warning(f"Revis?o recomendada: {result['review_reason']}")

    st.progress((idx + 1) / len(regions))
    st.write(f"Regi?o {idx + 1} de {len(regions)}")

    nav1, action1, action2, action3, nav2 = st.columns([2, 3, 3, 3, 2])

    with nav1:
        if idx > 0 and st.button("Anterior", key=f"prev_{idx}"):
            st.session_state.current_idx -= 1
            st.rerun()

    with action1:
        if st.button("Aprovar regi?o", key=f"approve_{idx}"):
            st.session_state.analysis["regions"][idx]["status"] = "approved"
            st.session_state.current_idx = min(idx + 1, len(regions) - 1)
            st.rerun()

    with action2:
        if st.button("Rejeitar regi?o", key=f"reject_{idx}"):
            st.session_state.analysis["regions"][idx]["status"] = "rejected"
            st.session_state.current_idx = min(idx + 1, len(regions) - 1)
            st.rerun()

    with action3:
        if st.button("Revisar regi?o", key=f"adjust_{idx}"):
            st.session_state.analysis["regions"][idx]["status"] = "needs_adjustment"
            st.session_state.current_idx = min(idx + 1, len(regions) - 1)
            st.rerun()

    with nav2:
        if idx < len(regions) - 1 and st.button("Pr?xima", key=f"next_{idx}"):
            st.session_state.current_idx += 1
            st.rerun()

    st.divider()
    st.subheader("Status da revis?o")
    approved = sum(1 for item in regions if item["status"] == "approved")
    rejected = sum(1 for item in regions if item["status"] == "rejected")
    needs_adjustment = sum(1 for item in regions if item["status"] == "needs_adjustment")
    pending = sum(1 for item in regions if item["status"] == "pending")

    summary1, summary2, summary3, summary4 = st.columns(4)
    summary1.metric("Aprovadas", approved)
    summary2.metric("Rejeitadas", rejected)
    summary3.metric("Ajuste solicitado", needs_adjustment)
    summary4.metric("Pendentes", pending)
