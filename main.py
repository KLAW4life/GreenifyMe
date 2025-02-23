import streamlit as st
import geocoder
import folium
import random
from streamlit_folium import st_folium

st.title("GreenifyMe: Microclimate Analysis Tool 🌱")

# Get the approximate location via IP address
g = geocoder.ip('me')

# -- Heat Map & User Location --

# Check if location data is available
if g.latlng:
    lat, lon = g.latlng
    st.success(f"📍 Your Location: Latitude: {lat}, Longitude: {lon}")
    # Coords will not be shown to users in final product

    # Create the map centered on the detected location
    map = folium.Map(location=[lat, lon], zoom_start=16, control_scale=True)

    # map = folium.Map(location=[lat, lon], zoom_start=12)

    # Add a marker at the user's location
    folium.Marker([lat, lon], popup="You are here!").add_to(map)

    # Display the map
    st_folium(map, width=700, height=500)
else:
    st.error("⚠️ Could not detect location. Please ensure you have an internet connection.")

# -- Greenify my surroundings --

st.header("🌿 Upload a Photo or Video for AI-Generated Greenery")

# User selects photo or video
file_type = st.radio("Choose file type to upload:", ("Photo", "Video"))

# Accepted file types
image_types = ["jpg", "jpeg", "png"]
video_types = ["mp4", "mov", "avi"]

# File uploader based on selection
uploaded_file = st.file_uploader(
    "Upload your file",
    type=image_types + video_types
)

# suggestions = [
#     "Add solar panels to your house",
#     "Plant new flowers"
# ]

# Process uploaded file
if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    if file_ext in image_types:
        st.image(uploaded_file, caption="Original Photo", use_container_width=True)

        # AI processing placeholder
        st.info("✅ AI Processing: Generating greenery-enhanced photo...")

        # Replace with AI-enhanced output
        st.success("🌱 AI-enhanced photo is ready!")
        st.image(uploaded_file, caption="AI-Enhanced Photo", use_container_width=True)  # Replace with AI result

    elif file_ext in video_types:
        st.video(uploaded_file)

        # AI processing placeholder
        st.info("✅ AI Processing: Generating greenery-enhanced video...")

        # Replace with AI-enhanced output
        st.success("🎥 AI-enhanced video is ready!")
        st.video(uploaded_file)  # Replace with AI-enhanced video

    else:
        st.error("❌ Unsupported file format.")