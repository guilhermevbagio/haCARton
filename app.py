import streamlit as st
import numpy as np
import cv2
import os
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

CLASS_NAMES = [
    "Água", "Floresta", "Grama", "Vegetação Alagada", "Lavoura",
    "Vegetação Rasteira", "Área Urbana", "Solo Exposto", "Neve"
]
CLASS_COLORS = [
    [47, 135, 224], [56, 168, 73], [191, 216, 106], [154, 194, 177],
    [229, 198, 129], [188, 177, 147], [196, 82, 102], [178, 164, 145], [242, 243, 244]
]

IMAGES_DIR = Path(__file__).parent / "data" / "images"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_images_from_folder():
    if not IMAGES_DIR.exists():
        return []
    files = sorted([
        f for f in IMAGES_DIR.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])
    images = []
    for f in files:
        img = cv2.imread(str(f))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return images


def generate_mock_image(seed):
    rng = np.random.RandomState(seed)
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)

    for _ in range(6):
        cx, cy = rng.randint(30, w - 30), rng.randint(30, h - 30)
        rw, rh = rng.randint(20, 80), rng.randint(20, 80)
        color = (
            int(rng.randint(30, 120)),
            int(rng.randint(80, 180)),
            int(rng.randint(20, 100)),
        )
        cv2.rectangle(img, (cx - rw, cy - rh), (cx + rw, cy + rh), color, -1)
        pts = rng.randint(0, min(h, w), (rng.randint(4, 8), 2)).astype(np.int32)
        cv2.fillPoly(img, [pts], color)

    for _ in range(8):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        radius = rng.randint(10, 40)
        color = (
            int(rng.randint(20, 100)),
            int(rng.randint(60, 150)),
            int(rng.randint(10, 80)),
        )
        cv2.circle(img, (cx, cy), radius, color, -1)

    noise = rng.normal(0, 12, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    return img


def create_work_units():
    classifications = [
        ("Floresta", 0.94),
        ("Lavoura", 0.87),
        ("Vegetação Rasteira", 0.72),
        ("Solo Exposto", 0.58),
        ("Área Urbana", 0.45),
    ]
    real_images = load_images_from_folder()
    units = []
    for i, (cls, conf) in enumerate(classifications):
        if i < len(real_images):
            image = real_images[i]
        else:
            image = generate_mock_image(seed=i * 42 + 7)
        units.append({
            "id": i + 1,
            "area_ha": 65000,
            "image": image,
            "classification": cls,
            "confidence": conf,
            "status": "pendente",
        })
    return units


def render_classification_overlay(image, class_idx):
    overlay = image.copy()
    color = CLASS_COLORS[class_idx % len(CLASS_COLORS)]
    mask = np.all(overlay > [20, 60, 10], axis=2)
    overlay[mask] = (np.array(overlay[mask]) * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    return overlay


if "work_units" not in st.session_state:
    st.session_state.work_units = create_work_units()
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

st.title("haCARton - Revisão de Classificação")
st.markdown("Blocos de trabalho classificados por modelo de IA.")

units = st.session_state.work_units
statuses = [u["status"] for u in units]
n_aceito = statuses.count("aceito")
n_rejeitado = statuses.count("rejeitado")
n_edicao = statuses.count("requer_edicao")
n_pendente = statuses.count("pendente")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Aceito", n_aceito)
col2.metric("Rejeitado", n_rejeitado)
col3.metric("Requer Edição", n_edicao)
col4.metric("Pendente", n_pendente)

st.progress((n_aceito + n_rejeitado + n_edicao) / len(units))

idx = st.session_state.current_idx
unit = units[idx]

class_idx = CLASS_NAMES.index(unit["classification"]) if unit["classification"] in CLASS_NAMES else 0
overlay_img = render_classification_overlay(unit["image"], class_idx)

status_label = {
    "pendente": "⏳ Pendente",
    "aceito": "✅ Aceito",
    "rejeitado": "❌ Rejeitado",
    "requer_edicao": "✏️ Requer Edição",
}[unit["status"]]

st.markdown(f"**Bloco #{unit['id']}** — Confiança: {unit['confidence']:.0%} — {status_label}")

col_img, col_class = st.columns(2)
with col_img:
    st.image(unit["image"], caption="Imagem Satelital", use_container_width=True)
with col_class:
    st.image(overlay_img, caption="Classificação", use_container_width=True)

st.progress((idx + 1) / len(units))

nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1, 2, 1, 1])

with nav1:
    if idx > 0:
        if st.button("← Anterior"):
            st.session_state.current_idx -= 1
            st.rerun()

if unit["status"] == "pendente":
    with nav3:
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("✓ Aceito"):
                st.session_state.work_units[idx]["status"] = "aceito"
                if idx < len(units) - 1:
                    st.session_state.current_idx += 1
                st.rerun()
        with b2:
            if st.button("✗ Rejeitado"):
                st.session_state.work_units[idx]["status"] = "rejeitado"
                if idx < len(units) - 1:
                    st.session_state.current_idx += 1
                st.rerun()
        with b3:
            if st.button("✎ Requer Edição"):
                st.session_state.work_units[idx]["status"] = "requer_edicao"
                if idx < len(units) - 1:
                    st.session_state.current_idx += 1
                st.rerun()
else:
    with nav3:
        if st.button("🔄 Resetar"):
            st.session_state.work_units[idx]["status"] = "pendente"
            st.rerun()

with nav5:
    if idx < len(units) - 1:
        if st.button("Próximo →"):
            st.session_state.current_idx += 1
            st.rerun()

if st.button("🔄 Reiniciar Tudo", use_container_width=True):
    st.session_state.work_units = create_work_units()
    st.session_state.current_idx = 0
    st.rerun()
