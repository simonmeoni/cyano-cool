import io
import itertools
import time

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


def assemble_images(input_images, dpi=300, width_cm=100, cyanotype_conversion=False):
    # Initialize list to store images
    px_cm = int(dpi / 2.54)
    img_stack = []
    img_big = []
    current_width = 0

    def normalize_image(img_stack, dimension):
        height_max = max(img.shape[dimension] for img in img_stack)
        for idx_img, img in enumerate(img_stack):
            img_stack[idx_img] = np.pad(
                img,
                ((0, height_max - img.shape[0]), (0, 0), (0, 0)),
                constant_values=255,
            )

    # Read each image and convert to numpy array
    for input_image in input_images:
        img_bytes = input_image.read()
        img = load_image(img_bytes)
        if cyanotype_conversion:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inverted = 255 - gray
            curve_data = load_acv_file(io.BytesIO(acv_bytes))
            img = apply_curve(inverted, curve_data)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.transpose(1, 0, 2) if img.shape[0] > img.shape[1] else img
        margin = int(px_cm / 4)
        img = np.pad(
            img,
            ((margin, margin), (margin, margin), (0, 0)),
            constant_values=255,
        )

        if current_width + img.shape[0] >= width_cm * px_cm:
            normalize_image(img_stack, 0)
            img_big.append(np.hstack(img_stack))
            img_stack = [img]
            current_width = 0
        else:
            img_stack.append(img)
            current_width += img.shape[1]

    if len(img_stack) > 0:
        normalize_image(img_stack, 0)
        img_big.append(np.hstack(img_stack))

    # Convert back to PIL Image, save to BytesIO object
    width_max = max(img_stack.shape[1] for img_stack in img_big)
    for idx_img_stack, img_stack in enumerate(img_big):
        img_big[idx_img_stack] = np.pad(
            img_stack,
            ((0, 0), (0, width_max - img_stack.shape[1]), (0, 0)),
            constant_values=255,
        )
    output_image_pil = Image.fromarray(np.uint8(np.vstack(img_big)))
    big_image_buffer = io.BytesIO()
    output_image_pil.save(big_image_buffer, format="TIFF", dpi=(dpi, dpi))
    big_image_buffer.seek(0)

    return {"content": big_image_buffer.getvalue()}


st.title("🎞️ Cyano-cool")

supported_type = ["jpg", "png", "tif", "jpeg", "tiff", "dng"]
uploaded_file = st.file_uploader("Choose an image file", type=supported_type)
uploaded_files = st.file_uploader(
    "Choose image files for conversion",
    type=supported_type,
    accept_multiple_files=True,
)
assemble_uploaded_files = st.file_uploader(
    "Choose image files to assemble",
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

    if st.button("🎊 Contact Negative Conversion"):
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
    if st.button("🪬 Cyanotype Folder Conversion"):
        contact_negative = convert_folder(uploaded_files, dpi=dpi)
        st.success("Images converted successfully!")
        st.download_button(
            label="Download Zip",
            data=contact_negative,
            file_name="output_images.zip",
            mime="application/zip",
        )
if assemble_uploaded_files:
    dpi_assemble = st.number_input(
        "🖨️ DPI: ", min_value=1, max_value=12000, value=300, step=1
    )
    width_cm = st.number_input(
        "🔛️ Width in cm: ", min_value=30, max_value=300, value=60, step=1
    )
    cyanotype_conversion = st.checkbox("🐈‍⬛ Cyanotype Conversion", value=False)
    if st.button("🤲 Assemble Negative for Print"):
        big_image = assemble_images(
            assemble_uploaded_files,
            dpi=dpi_assemble,
            width_cm=width_cm,
            cyanotype_conversion=cyanotype_conversion,
        )
        st.success("👐 Images assembled successfully!")
        st.download_button(
            label="Download assembled image",
            data=io.BytesIO(big_image["content"]),
            file_name=f"assembled_images_{time.strftime('%Y%m%d-%H%M%S')}.tif",
            mime=f"image/tiff",
        )
        result_image = Image.open(io.BytesIO(big_image["content"]))
        st.image(result_image, caption="Assembled Image")
