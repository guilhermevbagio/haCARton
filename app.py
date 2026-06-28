import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
from dw_processor import process_image, temporal_vote, render_overlay, CLASS_NAMES, CLASS_COLORS

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
st.markdown("Classify satellite imagery using local AI models.")

if "result" not in st.session_state:
    st.session_state.result = None
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "image_np" not in st.session_state:
    st.session_state.image_np = None

col_source, col_model = st.columns(2)
with col_source:
    source = st.radio("Data source", ["Upload Image", "Earth Engine (Sentinel-2)"], horizontal=True)
with col_model:
    model_choice = st.radio("Model", ["Dynamic World", "Prithvi", "Ensemble"], horizontal=True)

confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.0, 0.05,
                                 help="Filter out pixels below this confidence. 0 = show all.")

if source == "Upload Image":
    uploaded_file = st.file_uploader("Upload satellite image", type=["png", "jpg", "jpeg", "tif"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.image_np = np.array(image)
        st.image(image, width="stretch")

        if st.button("Process Image"):
            with st.spinner("Running Dynamic World classification..."):
                st.session_state.result = process_image(st.session_state.image_np)
                st.session_state.current_idx = 0
            st.rerun()

else:
    st.subheader("Earth Engine Location")
    col1, col2 = st.columns(2)
    with col1:
        lon = st.number_input("Longitude", value=-54.9732, format="%.4f")
    with col2:
        lat = st.number_input("Latitude", value=-19.9862, format="%.4f")

    col1, col2, col3 = st.columns(3)
    with col1:
        radius = st.slider("Radius (m)", 500, 5000, 2000, step=500)
    with col2:
        date_start = st.date_input("Start date", value=None)
    with col3:
        date_end = st.date_input("End date", value=None)

    col1, col2, col3 = st.columns(3)
    with col1:
        use_temporal = st.checkbox("Use temporal compositing", value=False)
    with col2:
        num_images = st.slider("Temporal images", 3, 15, 5, disabled=not use_temporal)
    with col3:
        temporal_method = st.selectbox("Composite method", ["median", "mean"], disabled=not use_temporal)

    col1, col2 = st.columns(2)
    with col1:
        fetch_ee = st.button("Fetch from Earth Engine")
    with col2:
        fetch_demo = st.button("Fetch Demo (cached)")

    if fetch_demo:
        demo_path = os.path.join(os.path.dirname(__file__), "demo_s2.npy")
        if os.path.exists(demo_path):
            s2_array = np.load(demo_path)
            display = s2_array[:, :, [2, 1, 0]]
            display = (display - display.min()) / (display.max() - display.min()) * 255
            st.session_state.image_np = display.astype(np.uint8)
            st.session_state.s2_array = s2_array
            st.image(st.session_state.image_np, width="stretch")
            st.success(f"Loaded demo image: {s2_array.shape}")
            st.rerun()
        else:
            st.error("Demo image not found. Run 'Fetch from Earth Engine' first to generate it.")

    if fetch_ee:
        try:
            import ee
            from ee_fetcher import initialize, fetch_sentinel2, fetch_temporal_stack, temporal_composite

            if not initialize():
                st.info("Opening Earth Engine authentication...")
                from ee_fetcher import authenticate
                authenticate()

            start_str = date_start.strftime('%Y-%m-%d') if date_start else '2024-01-01'
            end_str = date_end.strftime('%Y-%m-%d') if date_end else '2024-12-31'

            if use_temporal:
                with st.spinner(f"Fetching {num_images} temporal images for [{lon}, {lat}]..."):
                    arrays = fetch_temporal_stack(lon, lat, radius, start_str, end_str, max_images=num_images)
                    s2_array = temporal_composite(arrays, method=temporal_method)
            else:
                with st.spinner(f"Fetching Sentinel-2 imagery for [{lon}, {lat}]..."):
                    s2_array = fetch_sentinel2(lon, lat, radius, start_str, end_str)

            display = s2_array[:, :, [2, 1, 0]]
            display = (display - display.min()) / (display.max() - display.min()) * 255
            st.session_state.image_np = display.astype(np.uint8)
            st.session_state.s2_array = s2_array

            st.image(st.session_state.image_np, width="stretch")
            st.success(f"Loaded Sentinel-2 image: {s2_array.shape}")

        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.image_np is not None and st.button("Classify Image"):
        is_s2 = "s2_array" in st.session_state

        try:
            if model_choice == "Ensemble" and is_s2:
                with st.spinner("Running Ensemble (Dynamic World + Prithvi)..."):
                    from ensemble import ensemble_classify
                    st.session_state.result = ensemble_classify(
                        st.session_state.s2_array,
                        confidence_threshold=confidence_threshold
                    )
                    st.session_state.current_idx = 0
            elif model_choice == "Prithvi" and is_s2:
                with st.spinner("Running Prithvi classification..."):
                    from prithvi_processor import classify_with_prithvi
                    from dw_processor import CLASS_COLORS as DW_COLORS
                    prithvi_result = classify_with_prithvi(
                        st.session_state.s2_array,
                        confidence_threshold=confidence_threshold
                    )
                    color_overlay = DW_COLORS[prithvi_result["class_map"].clip(0)]
                    st.session_state.result = {
                        "class_map": prithvi_result["class_map"],
                        "confidence_map": prithvi_result["confidence_map"],
                        "color_overlay": color_overlay,
                        "regions": []
                    }
                    from dw_processor import extract_regions
                    st.session_state.result["regions"] = extract_regions(
                        prithvi_result["class_map"],
                        prithvi_result["confidence_map"]
                    )
                    st.session_state.current_idx = 0
            else:
                with st.spinner("Running Dynamic World classification..."):
                    st.session_state.result = process_image(
                        st.session_state.s2_array if is_s2 else st.session_state.image_np,
                        is_sentinel2=is_s2,
                        confidence_threshold=confidence_threshold
                    )
                    st.session_state.current_idx = 0
            st.rerun()
        except Exception as e:
            st.error(f"Classification failed: {e}")

if st.session_state.result and st.session_state.image_np is not None:
    result = st.session_state.result
    regions = result.get("regions", [])
    idx = st.session_state.current_idx

    st.divider()

    overlay_img = render_overlay(st.session_state.image_np, result)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(st.session_state.image_np, width="stretch")
    with col2:
        st.subheader("Classified")
        st.image(overlay_img, width="stretch")

    if regions:
        st.subheader("Classification Legend")
        legend_cols = st.columns(len(CLASS_NAMES))
        for i, (col, name) in enumerate(zip(legend_cols, CLASS_NAMES)):
            color_hex = "#{:02x}{:02x}{:02x}".format(*CLASS_COLORS[i])
            count = sum(1 for r in regions if r["class_idx"] == i)
            col.markdown(f'<div style="display:flex;align-items:center;gap:6px;">'
                        f'<div style="width:16px;height:16px;background:{color_hex};border:1px solid #333;"></div>'
                        f'<span style="font-size:12px;">{name} ({count})</span></div>',
                        unsafe_allow_html=True)

        if idx < len(regions):
            highlight = regions[idx]["id"]

            region = regions[idx]
            st.subheader(f"Region {idx + 1} of {len(regions)}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Class:** {region['class']}")
            with col2:
                st.write(f"**Confidence:** {region['confidence']:.0%}")
            with col3:
                st.write(f"**Area:** {region['area']} px²")

            status_colors = {
                "approved": "green",
                "rejected": "red",
                "needs_adjustment": "orange",
                "pending": "gray"
            }
            st.write(f"**Status:** :{status_colors[region['status']]}[{region['status']}]")

            st.progress((idx + 1) / len(regions))

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
    else:
        st.info("No regions found. Try adjusting the confidence threshold.")

    st.divider()
    st.subheader("Summary")
    approved = sum(1 for r in regions if r["status"] == "approved")
    rejected = sum(1 for r in regions if r["status"] == "rejected")
    pending = sum(1 for r in regions if r["status"] == "pending")

    col1, col2, col3 = st.columns(3)
    col1.metric("Approved", approved)
    col2.metric("Rejected", rejected)
    col3.metric("Pending", pending)
