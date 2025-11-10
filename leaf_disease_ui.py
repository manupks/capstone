import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import pickle

# ---------------------------------------------------------------
# ✅ 1. Helper Functions
# ---------------------------------------------------------------

def preprocess_image(image_path, target_size=(256, 256)):
    """Load and normalize image for segmentation."""
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_classification_image(image_path, target_size=(224, 224)):
    """Load and normalize image for disease classification."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def compute_severity(mask):
    """Compute disease/leaf pixel ratio."""
    disease_pixels = np.sum(mask == 2)
    leaf_pixels = np.sum((mask == 1) | (mask == 2))
    return disease_pixels / (leaf_pixels + disease_pixels) if (leaf_pixels + disease_pixels) > 0 else 0

# ---------------------------------------------------------------
# ✅ 2. Custom Losses / Metrics (for loading segmentation model)
# ---------------------------------------------------------------
NUM_CLASSES = 3
cce_loss_fn = tf.keras.losses.CategoricalCrossentropy()

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    dice = (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) + smooth
    )
    return 1 - tf.reduce_mean(dice)

def combined_loss(y_true, y_pred):
    return cce_loss_fn(y_true, y_pred) + dice_loss(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    dice = (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) + smooth
    )
    return tf.reduce_mean(dice)

def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)

# ---------------------------------------------------------------
# ✅ 3. Load Models (cached)
# ---------------------------------------------------------------
# @st.cache_resource
def load_unet_model():
    model = load_model(
        "segmentation_model/best_unet1/best_unet_model1.keras",
        custom_objects={
            'combined_loss': combined_loss,
            'dice_coef': dice_coef,
            'iou_coef': iou_coef
        },
        compile=False
    )
    return model
def load_classification_model():
    model = tf.keras.models.load_model("mobilenetv2_idg_base.h5")
    return model
# ---------------------------------------------------------------
# ✅ 4. Class Labels
# ---------------------------------------------------------------
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ---------------------------------------------------------------
# ✅ 5. Streamlit UI Layout
# ---------------------------------------------------------------
st.set_page_config(page_title="Leaf Disease Prediction + Segmentation", layout="wide")
st.title("🌿 Leaf Disease Detection, Segmentation & Severity Estimation")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save temporarily
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Display original image
    IMG_SEG_SIZE = (256, 256)
    IMG_CLASS_SIZE = (224, 224)
    img_display = load_img(temp_path, target_size=IMG_SEG_SIZE)
    img_display_np = img_to_array(img_display) / 255.0

    # -----------------------------------------------------------
    # 🔹 Disease Classification Prediction
    # -----------------------------------------------------------
    clf_model = load_classification_model()
    img_clf = preprocess_classification_image(temp_path, target_size=IMG_CLASS_SIZE)
    pred_probs = clf_model.predict(img_clf)
    pred_index = np.argmax(pred_probs, axis=1)[0]
    predicted_class = class_labels[pred_index]
    confidence = pred_probs[0][pred_index] * 100

    # -----------------------------------------------------------
    # 🔹 Segmentation Prediction
    # -----------------------------------------------------------
    unet_model = load_unet_model()
    img_seg = preprocess_image(temp_path, target_size=IMG_SEG_SIZE)
    pred_mask = unet_model.predict(img_seg)
    pred_mask_class = np.argmax(pred_mask, axis=-1)[0]
    severity = compute_severity(pred_mask_class)

    # -----------------------------------------------------------
    # 🔹 Sidebar Info
    # -----------------------------------------------------------
    st.sidebar.header("🔍 Model Info")
    st.sidebar.write("Predicted Class:", predicted_class)
    st.sidebar.write("Confidence:", f"{confidence:.2f}%")
    st.sidebar.write("Segmentation Output Shape:", pred_mask.shape)
    st.sidebar.write("Unique Mask Values:", np.unique(pred_mask_class))

    # -----------------------------------------------------------
    # 🔹 Visualization
    # -----------------------------------------------------------
    st.subheader("Results Visualization")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_display_np, caption="Original Image", use_container_width=True)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.imshow(pred_mask_class, cmap='jet')
        ax2.set_title("Predicted Segmentation Mask")
        ax2.axis('off')
        st.pyplot(fig2)
        plt.close(fig2)

    with col3:
        fig3, ax3 = plt.subplots()
        ax3.imshow(img_display_np)
        ax3.imshow(pred_mask[0, :, :, 2] if pred_mask.shape[-1] > 1 else pred_mask[0, :, :, 0], cmap='jet', alpha=0.5)
        ax3.set_title("Disease Overlay")
        ax3.axis('off')
        st.pyplot(fig3)
        plt.close(fig3)

    # -----------------------------------------------------------
    # 🔹 Text Results
    # -----------------------------------------------------------
    st.markdown(f"### 🩺 Predicted Disease: **{predicted_class}**")
    st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")
    st.markdown(f"### 🧬 Severity: **{severity * 100:.2f}%**")

    # -----------------------------------------------------------
    # 🔹 Optional: Show Segmentation Channels
    # -----------------------------------------------------------
    with st.expander("🔬 Show Segmentation Output Channels"):
        num_channels = pred_mask.shape[-1]
        for i in range(num_channels):
            fig, ax = plt.subplots()
            ax.imshow(pred_mask[0, :, :, i], cmap='jet')
            ax.set_title(f"Channel {i}")
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)

else:
    st.info("👆 Upload a leaf image to analyze disease type, segmentation, and severity.")
