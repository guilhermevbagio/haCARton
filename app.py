import streamlit as st
import numpy as np
import cv2
from PIL import Image
from mock_processor import process_image

st.set_page_config(page_title="haCARton - Satellite Review", layout="wide")

# Custom button colors
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
st.markdown("Upload satellite imagery and review AI segmentation/classification results.")

# Session state for results
if "results" not in st.session_state:
    st.session_state.results = []
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

# File upload
uploaded_file = st.file_uploader("Upload satellite image", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    if st.button("Process Image"):
        with st.spinner("Running segmentation and classification..."):
            st.session_state.results = process_image(image_np)
            st.session_state.current_idx = 0
        st.rerun()

# Review interface
if st.session_state.results:
    st.divider()
    st.subheader("Review Results")
    
    results = st.session_state.results
    idx = st.session_state.current_idx
    
    if idx < len(results):
        result = results[idx]
        
        # Overlay polygon on image
        overlay = image_np.copy()
        mask = result["mask"]
        
        # Handle RGBA images by converting to RGB
        if overlay.shape[2] == 4:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGBA2RGB)
            image_display = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        else:
            image_display = image_np.copy()
        
        # Color based on status
        if result["status"] == "approved":
            color = [0, 200, 0]
        elif result["status"] == "rejected":
            color = [200, 0, 0]
        elif result["status"] == "needs_adjustment":
            color = [200, 200, 0]
        else:
            color = [0, 100, 255]
        
        overlay[mask > 0] = color
        blended = cv2.addWeighted(image_display, 0.6, overlay, 0.4, 0)
        
        # Full width image
        st.image(blended, caption=f"Polygon {result['id']}", use_container_width=True)
        
        # Info row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Classification:** {result['class']}")
        with col2:
            st.write(f"**Confidence:** {result['confidence']:.0%}")
        with col3:
            st.write(f"**Status:** {result['status']}")
        
        # Progress
        st.progress((idx + 1) / len(results))
        st.write(f"Polygon {idx + 1} of {len(results)}")
        
        # Navigation with actions in the middle
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
                st.session_state.results[idx]["status"] = "approved"
                st.session_state.current_idx += 1
                st.rerun()
        
        with col3:
            if st.button("✗ Reject", key=f"reject_{idx}"):
                st.session_state.results[idx]["status"] = "rejected"
                st.session_state.current_idx += 1
                st.rerun()
        
        with col4:
            if st.button("⚠ Adjust", key=f"adjust_{idx}"):
                st.session_state.results[idx]["status"] = "needs_adjustment"
                st.session_state.current_idx += 1
                st.rerun()
        
        with spacer2:
            pass
        
        with col5:
            if idx < len(results) - 1:
                if st.button("Next →"):
                    st.session_state.current_idx += 1
                    st.rerun()
    
    # Summary
    st.divider()
    st.subheader("Summary")
    approved = sum(1 for r in results if r["status"] == "approved")
    rejected = sum(1 for r in results if r["status"] == "rejected")
    pending = sum(1 for r in results if r["status"] == "pending")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Approved", approved)
    col2.metric("Rejected", rejected)
    col3.metric("Pending", pending)