import streamlit as st
import numpy as np
import cv2
from PIL import Image
from dw_processor import process_image, render_overlay, CLASS_NAMES, CLASS_COLORS

st.set_page_config(page_title="haCARton - Satellite Review", layout="wide")

st.markdown("""
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
    .stImage img {
        max-height: 70vh;
        object-fit: contain;
    }
</style>
""", unsafe_allow_html=True)

st.title("haCARton - Satellite Image Review")
st.markdown("Upload satellite imagery and review Dynamic World classification results.")

if "result" not in st.session_state:
    st.session_state.result = None
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "image_np" not in st.session_state:
    st.session_state.image_np = None

uploaded_file = st.file_uploader("Upload satellite image", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.session_state.image_np = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")

    if st.button("Process Image"):
        with st.spinner("Running Dynamic World classification..."):
            st.session_state.result = process_image(st.session_state.image_np)
            st.session_state.current_idx = 0
        st.rerun()

if st.session_state.result and st.session_state.image_np is not None:
    result = st.session_state.result
    regions = result["regions"]
    idx = st.session_state.current_idx

    st.divider()

    if regions:
        # Class distribution
        st.subheader("Classification Legend")
        legend_cols = st.columns(len(CLASS_NAMES))
        for i, (col, name) in enumerate(zip(legend_cols, CLASS_NAMES)):
            color_hex = "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS[i])
            count = sum(1 for r in regions if r["class_idx"] == i)
            col.markdown(f'<div style="display:flex;align-items:center;gap:6px;">'
                        f'<div style="width:16px;height:16px;background:{color_hex};border:1px solid #333;"></div>'
                        f'<span style="font-size:12px;">{name} ({count})</span></div>',
                        unsafe_allow_html=True)

        # Show overlay
        st.subheader("Classification Overlay")
        highlight = regions[idx]["id"] if idx < len(regions) else None
        overlay_img = render_overlay(st.session_state.image_np, result, highlight)
        st.image(overlay_img, width="stretch")

        # Region info
        region = regions[idx]
        st.subheader(f"Region {idx + 1} of {len(regions)}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Class:** {region['class']}")
        with col2:
            st.write(f"**Confidence:** {region['confidence']:.0%}")
        with col3:
            st.write(f"**Area:** {region['area']} px²")

        # Status
        status_colors = {
            "approved": "green",
            "rejected": "red",
            "needs_adjustment": "orange",
            "pending": "gray"
        }
        st.write(f"**Status:** :{status_colors[region['status']]}[{region['status']}]")

        # Progress
        st.progress((idx + 1) / len(regions))

        # Navigation
        col1, spacer1, col2, col3, col4, spacer2, col5 = st.columns([2, 1, 3, 3, 3, 1, 2])

        with col1:
            if idx > 0:
                if st.button("← Prev"):
                    st.session_state.current_idx -= 1
                    st.rerun()

        with spacer1:
            pass

        with col2:
            if st.button("✓ Approve", key=f"approve_{idx}"):
                st.session_state.result["regions"][idx]["status"] = "approved"
                if idx < len(regions) - 1:
                    st.session_state.current_idx += 1
                st.rerun()

        with col3:
            if st.button("✗ Reject", key=f"reject_{idx}"):
                st.session_state.result["regions"][idx]["status"] = "rejected"
                if idx < len(regions) - 1:
                    st.session_state.current_idx += 1
                st.rerun()

        with col4:
            if st.button("⚠ Adjust", key=f"adjust_{idx}"):
                st.session_state.result["regions"][idx]["status"] = "needs_adjustment"
                if idx < len(regions) - 1:
                    st.session_state.current_idx += 1
                st.rerun()

        with spacer2:
            pass

        with col5:
            if idx < len(regions) - 1:
                if st.button("Next →"):
                    st.session_state.current_idx += 1
                    st.rerun()

    # Summary
    st.divider()
    st.subheader("Summary")
    approved = sum(1 for r in regions if r["status"] == "approved")
    rejected = sum(1 for r in regions if r["status"] == "rejected")
    pending = sum(1 for r in regions if r["status"] == "pending")

    col1, col2, col3 = st.columns(3)
    col1.metric("Approved", approved)
    col2.metric("Rejected", rejected)
    col3.metric("Pending", pending)
