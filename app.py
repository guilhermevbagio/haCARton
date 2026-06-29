import streamlit as st
import cv2
import numpy as np
from pathlib import Path

st.set_page_config(page_title="haCARton - Revisão de Imagens Satelitais", layout="wide")

st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) > div > button {
        background-color: #28a745 !important;
        color: white !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) > div > button {
        background-color: #dc3545 !important;
        color: white !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) > div > button {
        background-color: #ffc107 !important;
        color: black !important;
    }
    div[data-testid="stImage"] img {
        max-height: 40vh;
        object-fit: contain;
    }
</style>
""", unsafe_allow_html=True)

IMAGES_DIR = Path(__file__).parent / "data" / "images"


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_pairs():
    cru_files = sorted(IMAGES_DIR.glob("*cru.jpg"))
    pairs = []
    for cru_path in cru_files:
        num = cru_path.stem.replace("cru", "")
        classi_path = IMAGES_DIR / f"{num}classi.png"
        if classi_path.exists():
            cru = load_image(cru_path)
            classi = load_image(classi_path)
            if cru is not None and classi is not None:
                pairs.append({
                    "id": int(num),
                    "cru": cru,
                    "classi": classi,
                    "confidence": round(0.95 - (int(num) - 2) * 0.1, 2),
                    "status": "pendente",
                })
    return sorted(pairs, key=lambda x: x["confidence"], reverse=True)


if "pairs" not in st.session_state:
    st.session_state.pairs = load_pairs()
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

st.title("haCARton - Revisão de Classificação")

pairs = st.session_state.pairs
statuses = [p["status"] for p in pairs]
n_aceito = statuses.count("aceito")
n_rejeitado = statuses.count("rejeitado")
n_edicao = statuses.count("requer_edicao")
n_pendente = statuses.count("pendente")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Aceito", n_aceito)
col2.metric("Rejeitado", n_rejeitado)
col3.metric("Requer Edição", n_edicao)
col4.metric("Pendente", n_pendente)

st.progress((n_aceito + n_rejeitado + n_edicao) / len(pairs))

idx = st.session_state.current_idx
pair = pairs[idx]

status_label = {
    "pendente": "⏳ Pendente",
    "aceito": "✅ Aceito",
    "rejeitado": "❌ Rejeitado",
    "requer_edicao": "✏️ Requer Edição",
}[pair["status"]]

st.markdown(f"**Bloco #{pair['id']}** — Confiança: {pair['confidence']:.0%} — {status_label}")

col_cru, col_classi = st.columns(2)
with col_cru:
    st.image(pair["cru"], caption="Imagem Cru", use_container_width=True)
with col_classi:
    st.image(pair["classi"], caption="Classificação", use_container_width=True)

st.progress((idx + 1) / len(pairs))

nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1, 2, 1, 1])

with nav1:
    if idx > 0:
        if st.button("← Anterior"):
            st.session_state.current_idx -= 1
            st.rerun()

if pair["status"] == "pendente":
    with nav3:
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("✓ Aceito"):
                st.session_state.pairs[idx]["status"] = "aceito"
                if idx < len(pairs) - 1:
                    st.session_state.current_idx += 1
                st.rerun()
        with b2:
            if st.button("✗ Rejeitado"):
                st.session_state.pairs[idx]["status"] = "rejeitado"
                if idx < len(pairs) - 1:
                    st.session_state.current_idx += 1
                st.rerun()
        with b3:
            if st.button("✎ Requer Edição"):
                st.session_state.pairs[idx]["status"] = "requer_edicao"
                if idx < len(pairs) - 1:
                    st.session_state.current_idx += 1
                st.rerun()
else:
    with nav3:
        if st.button("🔄 Resetar"):
            st.session_state.pairs[idx]["status"] = "pendente"
            st.rerun()

with nav5:
    if idx < len(pairs) - 1:
        if st.button("Próximo →"):
            st.session_state.current_idx += 1
            st.rerun()

if st.button("🔄 Reiniciar Tudo", use_container_width=True):
    st.session_state.pairs = load_pairs()
    st.session_state.current_idx = 0
    st.rerun()
