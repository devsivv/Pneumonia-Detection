# gradcam.py

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image


IMG_SIZE = (224, 224)


def preprocess_image(img_path):
    """
    Load and preprocess image for prediction.

    IMPORTANT:
    Do NOT divide by 255.
    The saved VGG16 model already contains:
    Rescaling(1./255)
    """

    img = image.load_img(
        img_path,
        target_size=IMG_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


def build_cam_model(model):
    """
    Build Grad-CAM model from loaded VGG16 model.

    Compatible with:
    Sequential
        ├─ Rescaling
        ├─ VGG16
        ├─ GAP
        ├─ Dense
        ├─ Dropout
        └─ Dense
    """

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = model.layers[0](inputs)
    vgg_out = model.layers[1](x)

    x = model.layers[2](vgg_out)
    x = model.layers[3](x)
    x = model.layers[4](x)
    outputs = model.layers[5](x)

    cam_model = tf.keras.Model(
        inputs=inputs,
        outputs=[vgg_out, outputs]
    )

    return cam_model


def make_gradcam_heatmap(img_array, cam_model):
    """
    Generate Grad-CAM heatmap.
    """

    with tf.GradientTape() as tape:

        conv_outputs, predictions = cam_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_val = tf.reduce_max(heatmap)

    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()


def overlay_gradcam(img_path, heatmap):
    """
    Create Grad-CAM overlay image.
    """

    img = cv2.imread(img_path)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap_colored = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap_colored = cv2.cvtColor(
        heatmap_colored,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        img,
        0.6,
        heatmap_colored,
        0.4,
        0
    )

    return overlay


def generate_gradcam(model, image_path):
    """
    Complete Grad-CAM pipeline.

    Returns:
        overlay_image (RGB numpy array)
    """

    cam_model = build_cam_model(model)

    img_array = preprocess_image(image_path)

    heatmap = make_gradcam_heatmap(
        img_array,
        cam_model
    )

    overlay = overlay_gradcam(
        image_path,
        heatmap
    )

    return overlay