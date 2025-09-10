import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource  # caches the model so it loads only once
def load_model():
    return tf.keras.models.load_model("cat_dog_cifar10.h5")

model = load_model()

# Prediction function
def predict(img):
    img = img.resize((32, 32))  # CIFAR-10 size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    return "🐶 Dog" if pred > 0.5 else "🐱 Cat"

# Streamlit UI
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image and let the model classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")
    label = predict(image)
    st.success(f"Prediction: **{label}**")
