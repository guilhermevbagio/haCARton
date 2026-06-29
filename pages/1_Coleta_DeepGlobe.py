import streamlit as st

from streamlit_app_utils import (
    SEGMENTER_OPTIONS,
    apply_global_styles,
    ensure_session_defaults,
    render_deepglobe_mode,
)

st.set_page_config(page_title="haCARton - Coleta DeepGlobe", layout="wide")
apply_global_styles()
ensure_session_defaults()

st.title("haCARton - Coleta DeepGlobe")
st.markdown(
    "Revise imagens em lote a partir de uma pasta local do DeepGlobe e gere "
    "novos exemplos rotulados para o validador automático."
)

segmenter_label = st.selectbox(
    "Modo de processamento",
    options=list(SEGMENTER_OPTIONS.keys()),
    help=(
        "O modo experimental RSAM-Seg está preparado na arquitetura, mas o "
        "protótipo atual ainda usa fallback heurístico para manter a demo estável."
    ),
)
segmenter_key = SEGMENTER_OPTIONS[segmenter_label]

render_deepglobe_mode(segmenter_key)
