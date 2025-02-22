import streamlit as st
import geocoder
import folium
from streamlit_folium import st_folium

st.title("GreenifyMe: Microclimate Analysis Tool 🌱")

# Get the approximate location via IP address
g = geocoder.ip('me')

# Check if location data is available
if g.latlng:
    lat, lon = g.latlng
    st.success(f"📍 Your Location: Latitude: {lat}, Longitude: {lon}")
    # coords will not be shown to users in final product

    # Create the map centered on the detected location
    map = folium.Map(location=[lat, lon], zoom_start=16, control_scale=True)

    # map = folium.Map(location=[lat, lon], zoom_start=12)

    # Add a marker at the user's location
    folium.Marker([lat, lon], popup="You are here!").add_to(map)

    # Display the map
    st_folium(map, width=700, height=500)
else:
    st.error("⚠️ Could not detect location. Please ensure you have an internet connection.")
