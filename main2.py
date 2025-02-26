import numpy as np
import streamlit as st
# import geocoder
import folium
# from moviepy import VideoFileClip
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import random
import os
# import base64
from PIL import Image
# import io
# from streamlit.components.v1 import html

# --- Page Config ---
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

    .subtitle {
        color: #166534;
        font-size: 1.2rem;
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
    <div class="subtitle">Microclimate Analysis Tool</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -- Temperature Data (hardcoded for the purpose of saving time) --

@st.cache_data
def generate_heatmap_data():
    lat_min, lat_max = 25.70, 25.82  # Miami area
    lon_min, lon_max = -80.25, -80.15
    grid_size = 0.001  # Smaller grid for more density

    lat_values = np.random.uniform(lat_min, lat_max, 500)  # Generate 500 random latitude points
    lon_values = np.random.uniform(lon_min, lon_max, 500)  # Generate 500 random longitude points

    heatmap_data = []
    for lat, lon in zip(lat_values, lon_values):
        intensity = random.uniform(28, 35)  # Fake temperature values
        heatmap_data.append([lat, lon, intensity])

    return heatmap_data

if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = generate_heatmap_data()

def create_heatmap():
    heatmap_data = st.session_state.heatmap_data
    map_disp = folium.Map(location=[25.7617, -80.1918], zoom_start=13, control_scale=True)
    HeatMap(heatmap_data, radius=15, blur=10, min_opacity=0.5).add_to(map_disp)
    return map_disp

# --- Main Content ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Heatmap of Your Area</div>', unsafe_allow_html=True)

    # Create and display the heatmap
    heatmap = create_heatmap()

    st_folium(heatmap, height=400)

    st.markdown('</div>', unsafe_allow_html=True)

suggestions = [
    "Add solar panels to roof of the building",
    "Plant wildflower patches",
    "Construct sidewalks or bicycle lanes to decrease carbon emissions",
    "Implement recycling bins around the area",
    "Plant native trees and shrubs"
]

with col2:
    # Upload Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Upload a Photo for AI-Generated Greenery</div>', unsafe_allow_html=True)

    photo_file = st.file_uploader("Upload a Photo", type=["jpg", "jpeg", "png"], key="photo")
    if photo_file:
        image = Image.open(photo_file)
        st.image(image, caption="Uploaded Photo", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # st.markdown('</div>', unsafe_allow_html=True)

    # Simulation Results Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_title, col_button = st.columns([2, 1])
    with col_title:
        st.markdown('<div class="card-title">Simulation Results</div>', unsafe_allow_html=True)
    with col_button:
        simulate_button = st.button("Simulate")

    if simulate_button:
        st.info("Running simulation...")

        # Show pre-AI and post-AI images as an example
        pre_ai_image = Image.open("Miami_Rundown.jpg")
        post_ai_image = Image.open("Rundown_AI_Gen_Image.jpg")

        st.subheader("🌍 Example of AI-enhanced Greenery")

        col1, col2 = st.columns(2)

        with col1:
            st.image(pre_ai_image, caption="Before AI Enhancement", use_container_width=True)

        with col2:
            st.image(post_ai_image, caption="After AI Enhancement", use_container_width=True)

        st.subheader("🌱 How to Further Greenify This Area")
        num_suggestions = random.randint(1, len(suggestions))  # Choose random number of suggestions
        selected_suggestions = random.sample(suggestions, num_suggestions)

        for suggestion in selected_suggestions:
            st.write(f"- {suggestion}")
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

    # Simulation Results Section
    # st.markdown('<div class="card">', unsafe_allow_html=True)
    # col_title, col_button = st.columns([2, 1])
    # with col_title:
    #     st.markdown('<div class="card-title">Simulation Results</div>', unsafe_allow_html=True)
    # with col_button:
    #     simulate_button = st.button("Simulate")
    #
    # if simulate_button:
    #     st.info("Running simulation...")
    #
    #     # After simulation, generate and display suggestions
    #     num_suggestions = random.randint(1, len(suggestions))  # Choose random number of suggestions
    #     selected_suggestions = random.sample(suggestions, num_suggestions)
    #
    #     st.subheader("🌍 How to Further Greenify This Area")
    #     for suggestion in selected_suggestions:
    #         st.write(f"- {suggestion}")
    #
    # else:
    #     st.markdown("""
    #         <div style="
    #             height: 200px;
    #             background-color: #f0fdf4;
    #             border-radius: 0.5rem;
    #             display: flex;
    #             align-items: center;
    #             justify-content: center;
    #             color: #166534;
    #         ">
    #             No simulation data available
    #         </div>
    #         """, unsafe_allow_html=True)
    #
    # st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("<br>", unsafe_allow_html=True)
