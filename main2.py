import streamlit as st
import geocoder
import folium
from streamlit_folium import st_folium
import base64
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="GreenifyMe",
    page_icon="🌱",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Main styling */
    .stApp {
        background-color: white;
    }

    /* Header styling */
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }

    .logo {
        color: #15803d;
        font-size: 1.5rem;
        font-weight: bold;
        text-decoration: none;
    }

    .nav-links {
        display: flex;
        gap: 2rem;
    }

    .nav-link {
        color: #166534;
        text-decoration: none;
        transition: color 0.2s;
    }

    .nav-link:hover {
        color: #15803d;
    }

    /* Card styling */
    .card {
        background-color: white;
        border: 2px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 1.25rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }

    /* Upload area styling */
    .upload-area {
        border: 2px dashed #e5e7eb;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        background-color: #f9fafb;
    }

    /* Button styling */
    .stButton > button {
        background-color: #15803d;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
    }

    .stButton > button:hover {
        background-color: #166534;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header">
    <div class="logo">GreenifyMe</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main Content ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Interactive Heatmap of Your Area</div>', unsafe_allow_html=True)

    # Get location and create map
    g = geocoder.ip('me')
    if g.latlng:
        lat, lon = g.latlng
        map = folium.Map(location=[lat, lon], zoom_start=16, control_scale=True)
        folium.Marker([lat, lon], popup="You are here!").add_to(map)
        st_folium(map, height=400)
    else:
        st.error("⚠️ Could not detect location. Please ensure you have an internet connection.")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Upload Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Upload Media</div>', unsafe_allow_html=True)

    # Tabs for Photo/Video
    tab1, tab2 = st.tabs(["📸 Photo", "🎥 Video"])

    with tab1:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        photo_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"], key="photo")
        if photo_file:
            image = Image.open(photo_file)
            st.image(image, caption="Uploaded Photo", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="video")
        if video_file:
            st.video(video_file)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Simulation Results Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_title, col_button = st.columns([2, 1])
    with col_title:
        st.markdown('<div class="card-title">Simulation Results</div>', unsafe_allow_html=True)
    with col_button:
        simulate_button = st.button("Simulate")

    if simulate_button:
        st.info("Running simulation...")
        # Add simulation logic here
    else:
        st.markdown("""
        <div style="
            height: 200px;
            background-color: #f0fdf4;
            border-radius: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #166534;
        ">
            No simulation data available
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("<br>", unsafe_allow_html=True)