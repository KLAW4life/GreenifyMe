import streamlit as st
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import torch
import cv2
import os

# --------------------------- Configuration --------------------------- #

# Prompt to guide the inpainting
DEFAULT_PROMPT = (
    "Transform the image by filling empty spaces with green environments or "
    "infrastructures that benefit climate change."
)

# Hugging Face Authentication Token
HUGGINGFACE_TOKEN = ""  # Replace with your token

# Device configuration
DEVICE = "cpu"  # Change to "cuda" if you have a compatible GPU


# --------------------------- Helper Functions --------------------------- #

def load_inpainting_model():
    """Load the Stable Diffusion 2 Inpainting model."""
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        use_auth_token=HUGGINGFACE_TOKEN,
    ).to(DEVICE)
    return pipe


def resize_image(image, max_dim=512):
    """Resize the image to a maximum dimension to optimize processing time."""
    width, height = image.size
    max_side = max(width, height)
    if max_side > max_dim:
        scale = max_dim / max_side
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size)
    return image


def detect_empty_spaces(image):
    """Automatically detect empty spaces in the image and create a mask."""
    # Convert PIL image to NumPy array and then to grayscale
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Invert edges to get edges as black on white background
    edges_inv = cv2.bitwise_not(edges)

    # Threshold to get binary image
    _, thresh = cv2.threshold(edges_inv, 200, 255, cv2.THRESH_BINARY)

    # Find areas with low edge density
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Optionally, you can further refine the mask to exclude small areas
    # mask = cv2.erode(mask, kernel, iterations=1)
    # mask = cv2.dilate(mask, kernel, iterations=1)

    # Convert mask to PIL Image
    mask_image = Image.fromarray(mask)

    return mask_image


def inpaint_image(pipe, image, mask, prompt):
    """Perform the inpainting operation."""
    # Ensure images are in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    if mask.mode != "RGB":
        mask = mask.convert("RGB")

    # Run the inpainting pipeline
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=7.5,
        num_inference_steps=30,
    )
    return result.images[0]


# --------------------------- Streamlit App --------------------------- #

def main():
    st.title("🌍 Climate Action Image Transformer")
    st.write("Envision greener environments by transforming empty spaces in your photos.")

    # Sidebar for options
    st.sidebar.title("🛠️ Options")

    # Prompt customization
    prompt = st.sidebar.text_area("Enter your custom prompt:", value=DEFAULT_PROMPT)

    # Upload image
    uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

        # Resize image
        image_resized = resize_image(image)

        # Detect empty spaces and create mask
        with st.spinner("Detecting empty spaces in the image..."):
            mask = detect_empty_spaces(image_resized)
        st.image(mask, caption="Generated Mask", use_container_width=True)

        # Transform button
        if st.button("🌱 Transform Image"):
            with st.spinner("Processing... This may take a few moments ⏳"):
                # Load the model
                pipe = load_inpainting_model()

                # Perform inpainting
                output_image = inpaint_image(pipe, image_resized, mask, prompt)

            st.success("✨ Transformation Complete!")
            st.image(output_image, caption="Transformed Image", use_container_width=True)
    else:
        st.info("Please upload an image to begin.")


if __name__ == "__main__":
    main()