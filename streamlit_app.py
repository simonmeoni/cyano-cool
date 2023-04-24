import io
import os
import zipfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

acv_bytes = open("32419-cyanotype-curve.acv", "rb").read()


def load_acv_file(acv_file):
    data = acv_file.read()
    return np.frombuffer(data, dtype=">h", offset=2)


def apply_curve(image, curve):
    lut = np.interp(np.arange(256), np.linspace(0, 255, len(curve)), curve).astype(
        "uint8"
    )
    return cv2.LUT(image, lut)


def load_image(input_image_bytes):
    return cv2.imdecode(np.frombuffer(input_image_bytes, np.uint8), cv2.IMREAD_COLOR)


def convert_to_contact_negative(img_bytes, extension):
    # Load the input image
    img = load_image(img_bytes)
    if img is None:
        return {"error": "Cannot read the input image."}
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert the image
    inverted = 255 - gray
    # Load the curve data from the .acv file
    curve_data = load_acv_file(io.BytesIO(acv_bytes))
    # Apply the curve adjustment
    adjusted = apply_curve(inverted, curve_data)
    # Save the output image
    is_success, buffer = cv2.imencode(f".{extension}", adjusted)
    if is_success:
        return {"content": buffer.tobytes()}
    else:
        return {"error": "Cannot save the output image."}


def apply_brightness_contrast(img_array, brightness, contrast):
    img_array = np.int16(img_array)
    img_array = img_array * (contrast / 127 + 1) - contrast + brightness
    img_array = np.clip(img_array, 0, 255)
    return np.uint8(img_array)


def convert_folder(input_images):
    memfile = io.BytesIO()
    with zipfile.ZipFile(memfile, mode="w") as zf:
        for input_image in input_images:
            # Load the input image
            img_bytes = input_image.read()
            extension = input_image.name.split(".")[-1]
            buffer = convert_to_contact_negative(img_bytes, extension)['content']
            # Save the output image
            output_filename = f"output_{input_image.name}"

            with open(output_filename, mode="wb") as f:
                f.write(buffer)
            zf.write(output_filename, arcname=output_filename)

    return memfile.getvalue()

st.title("🎞️ Cyano-cool")

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "png", "tif", "jpeg", "tiff", "dng"],
)
uploaded_files = st.file_uploader(
    "Choose image files",
    type=["jpg", "png", "tif", "jpeg", "tiff", "dng"],
    accept_multiple_files=True,
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Add brightness and contrast sliders
    brightness = st.slider("🔆 Brightness", -100, 100, 0)
    contrast = st.slider("👁️‍🗨️ Contrast", -100, 100, 0)
    # Check if the sliders have changed and apply the adjustments
    if "last_brightness" not in st.session_state:
        st.session_state.last_brightness = brightness
    if "last_contrast" not in st.session_state:
        st.session_state.last_contrast = contrast
    if "adjusted_image" not in st.session_state:
        st.session_state.adjusted_image = img_array
    if (
        st.session_state.last_brightness != brightness
        or st.session_state.last_contrast != contrast
    ):
        img_array = apply_brightness_contrast(img_array, brightness, contrast)
        st.session_state.adjusted_image = img_array
        st.session_state.last_brightness = brightness
        st.session_state.last_contrast = contrast
    else:
        img_array = st.session_state.adjusted_image

    st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), caption="Adjusted Image")

    if st.button("Convert to Cyanotype"):
        # Save the adjusted image to a new buffer
        img_array = apply_brightness_contrast(image, brightness, contrast)
        adjusted_buffer = io.BytesIO()

        # Get the file extension and format from the uploaded file
        _, file_extension = os.path.splitext(uploaded_file.name)
        img_array = apply_brightness_contrast(img_array, brightness, contrast)
        Image.fromarray(img_array).save(adjusted_buffer, format="TIFF")
        adjusted_buffer.seek(0)

        # Send the adjusted image to the server
        contact_negative = convert_to_contact_negative(
            adjusted_buffer.getvalue(), file_extension
        )
        result_image = Image.open(io.BytesIO(contact_negative['content']))
        st.image(result_image, caption="Cyanotype Image")
        # Add a download button
        st.download_button(
            label="Download Cyanotype Image",
            data=io.BytesIO(contact_negative['content']),
            file_name=f"{uploaded_file.name.split('.')[0]}_cyanotype{file_extension}",
            mime=f"image/tiff",
        )

if uploaded_files:
    files = {
        f"input_image_{i}": (
            uploaded_file.name,
            Image.fromarray(load_image(uploaded_file.getvalue())).save(io.BytesIO(), format="TIFF"),
        )
        for i, uploaded_file in enumerate(uploaded_files)
    }
    if st.button("Convert Folder to Cyanotype"):
        contact_negative = convert_folder(uploaded_files)

        st.success("Images converted successfully!")
        st.download_button(
            label="Download Zip",
            data=contact_negative,
            file_name="output_images.zip",
            mime="application/zip",
        )
