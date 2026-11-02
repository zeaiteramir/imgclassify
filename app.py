import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

MODEL_DIR = Path("model")
KERNEL_SLUG = "amirzeaiter/intelimage-classification"
MODEL_FILENAME = "best_scene_model.keras"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
IMAGE_SIZE = (224, 224)
# Alphabetical order, matching the folder order used during training.
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


def ensure_model() -> Path:
    """Download the trained model from the Kaggle kernel if it is missing."""
    if not MODEL_PATH.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(MODEL_DIR)],
            check=True,
        )
    return MODEL_PATH


@st.cache_resource(show_spinner="Loading the model...")
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model(ensure_model(), compile=False)


def classify(model: tf.keras.Model, image: Image.Image) -> np.ndarray:
    """Return the probability of each class. The model already rescales pixels and applies softmax."""
    resized = image.convert("RGB").resize(IMAGE_SIZE)
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    return model.predict(batch, verbose=0)[0]


def confidence_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.50:
        return "Medium"
    return "Low"


st.set_page_config(page_title="Scene Classifier", page_icon="🖼️")
st.title("🖼️ Scene Classifier")
st.write(
    "Upload a photo and I'll tell you what kind of scene it shows. "
    "I can recognize: " + ", ".join(CLASS_NAMES) + "."
)

uploaded = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("👆 Upload a photo to get started.")
    st.stop()

image = Image.open(uploaded)
model = load_model()

with st.spinner("Looking at your photo..."):
    probabilities = classify(model, image)

best_index = int(np.argmax(probabilities))
best_label = CLASS_NAMES[best_index]
confidence = float(probabilities[best_index])
level = confidence_level(confidence)
message = (
    f"I think this is a **{best_label}**. "
    f"Confidence: **{confidence:.1%}** ({level} confidence)."
)

left, right = st.columns(2)
with left:
    st.image(image, caption="Your photo")
with right:
    if level == "Low":
        st.warning("🤔 " + message + " The model is not sure about this one.")
    else:
        st.success("✅ " + message)
    st.write("Confidence for every class:")
    st.bar_chart(pd.DataFrame({"confidence": probabilities}, index=CLASS_NAMES))
