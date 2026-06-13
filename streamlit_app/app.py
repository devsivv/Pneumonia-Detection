from huggingface_hub import hf_hub_download
import tensorflow as tf
import streamlit as st
import numpy as np
import os

from tensorflow.keras.preprocessing import image
from PIL import Image

# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Pneumonia Detection",
    layout="centered"
)
# ==================================================
# Load Model
# ==================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id="devsivvHF/pneumonia-detection-models",
        filename="vgg16_pneumonia.keras"
    )

    model = tf.keras.models.load_model(model_path)

    return model


model = load_model()

# ==================================================
# Sidebar
# ==================================================

with st.sidebar:

    st.title("Model Information")

    st.write("**Model:** VGG16 Transfer Learning")
    st.write("**Test Accuracy:** 85.10%")
    st.write("**Dataset:** 5,856 Chest X-rays")
    st.write("**Classes:** NORMAL, PNEUMONIA")

    st.divider()

    st.subheader("Classes")

    st.write("• NORMAL")
    st.write("• PNEUMONIA")

    st.divider()

    st.write(
        """
        This application uses a deep learning model
        trained on chest X-ray images to detect
        pneumonia.
        """
    )

# ==================================================
# Header
# ==================================================

st.title("Pneumonia Detection")

st.markdown(
    "Chest X-ray classification using Deep Learning and Transfer Learning (VGG16)."
)
st.divider()

# ==================================================
# Upload Section
# ==================================================

col1, col2 = st.columns([2, 1])

with col1:

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type=["jpg", "jpeg", "png"]
    )

with col2:

    st.info(
        """
        **Supported Formats**

        • JPG  
        • JPEG  
        • PNG
        """
    )

# ==================================================
# Prediction
# ==================================================

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    # Display image centered
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.image(
            img,
            caption="Uploaded Chest X-ray",
            width=400
        )

    # Preprocessing
    resized_img = img.resize((224, 224))

    img_array = image.img_to_array(resized_img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    probability = prediction[0][0]

    if probability >= 0.5:

        label = "PNEUMONIA"
        confidence = float(probability)

    else:

        label = "NORMAL"
        confidence = float(1 - probability)

    st.divider()

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Class",
            value=label
        )

    with col2:
        st.metric(
            label="Confidence",
            value=f"{confidence*100:.2f}%"
        )

    st.progress(confidence)

    if label == "NORMAL":

        st.success(
            "No signs of pneumonia detected."
        )

    else:

        st.error(
            "Pneumonia detected."
        )

# ==================================================
# Footer
# ==================================================

st.divider()

st.divider()

st.caption(
    "Developed using TensorFlow, Keras and Streamlit"
)

st.markdown(
    """
    **Developer:** Shivam Dubey  
    **GitHub:** [github.com/devsivv](https://github.com/devsivv)
    """
)

