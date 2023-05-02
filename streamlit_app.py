import io
import math
import os
import zipfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

acv_bytes = open("32419-cyanotype-curve.acv", "rb").read()


def load_acv_file(acv_file):
    return np.frombuffer(acv_file.read(), dtype=">h", offset=2)


def apply_curve(image, curve):
    lut = np.interp(np.arange(256), np.linspace(0, 255, len(curve)), curve).astype(
        "uint8"
    )
    return cv2.LUT(image, lut)


def load_image(input_image_bytes):
    return cv2.imdecode(np.frombuffer(input_image_bytes, np.uint8), cv2.IMREAD_COLOR)


def convert_to_contact_negative(img_bytes, extension, dpi):
    img = load_image(img_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    curve_data = load_acv_file(io.BytesIO(acv_bytes))
    adjusted = apply_curve(inverted, curve_data)

    adjusted_buffer = io.BytesIO()
    Image.fromarray(adjusted).save(adjusted_buffer, format="TIFF", dpi=(dpi, dpi))
    adjusted_buffer.seek(0)

    return (
        {"content": adjusted_buffer.getvalue()}
        if adjusted_buffer
        else {"error": "Cannot save the output image."}
    )


def apply_brightness_contrast(img_array, brightness, contrast):
    img_array = np.int16(img_array)
    img_array = img_array * (contrast / 127 + 1) - contrast + brightness
    img_array = np.clip(img_array, 0, 255)
    return np.uint8(img_array)


def convert_folder(input_images, dpi=300):
    memfile = io.BytesIO()
    with zipfile.ZipFile(memfile, mode="w") as zf:
        for input_image in input_images:
            img_bytes = input_image.read()
            extension = input_image.name.split(".")[-1]
            buffer = convert_to_contact_negative(img_bytes, extension, dpi=dpi)[
                "content"
            ]
            output_filename = f"output_{input_image.name}"
            with open(output_filename, mode="wb") as f:
                f.write(buffer)
            zf.write(output_filename, arcname=output_filename)
            os.remove(output_filename)
    return memfile.getvalue()


st.title("🎞️ Cyano-cool")

supported_type = ["jpg", "png", "tif", "jpeg", "tiff", "dng"]
uploaded_file = st.file_uploader("Choose an image file", type=supported_type)
uploaded_files = st.file_uploader(
    "Choose image files",
    type=supported_type,
    accept_multiple_files=True,
)

if uploaded_file:
    dpi = st.number_input("🖨️ DPI: ", min_value=1, max_value=12000, value=300, step=1)
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    brightness = st.slider("🔆 Brightness", -100, 100, 0)
    contrast = st.slider("👁️‍🗨️ Contrast", -100, 100, 0)

    img_array = apply_brightness_contrast(img_array, brightness, contrast)
    st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), caption="Adjusted Image")

    if st.button("Contact Negative Conversion"):
        img_array = apply_brightness_contrast(image, brightness, contrast)
        adjusted_buffer = io.BytesIO()
        _, file_extension = os.path.splitext(uploaded_file.name)
        Image.fromarray(img_array).save(adjusted_buffer, format="TIFF", dpi=(dpi, dpi))
        adjusted_buffer.seek(0)
        contact_negative = convert_to_contact_negative(
            adjusted_buffer.getvalue(), file_extension, dpi
        )
        result_image = Image.open(io.BytesIO(contact_negative["content"]))
        st.image(result_image, caption="Cyanotype Image")

        st.download_button(
            label="Download Negative",
            data=io.BytesIO(contact_negative["content"]),
            file_name=f"{uploaded_file.name.split('.')[0]}_negative{file_extension}",
            mime=f"image/{file_extension}",
        )
if uploaded_files:
    dpi = st.number_input("🖨️ DPI: ", min_value=1, max_value=12000, value=300, step=1)
    if st.button("Cyanotype Folder Conversion"):
        contact_negative = convert_folder(uploaded_files, dpi=dpi)
        st.success("Images converted successfully!")
        st.download_button(
            label="Download Zip",
            data=contact_negative,
            file_name="output_images.zip",
            mime="application/zip",
        )
