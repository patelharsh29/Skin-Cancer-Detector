import streamlit as st

# Title
st.set_page_config(page_title="Skin Cancer Detector", page_icon="🩺", layout="centered")

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# loading model (cached for speed)
@st.cache_resource
def load_cnn_model():
    return load_model("skin_cancer_cnn.h5")

model = load_cnn_model()


# preprocessing and predicting 
def predict_skin_lesion(image_file, model, target_size=(224, 224), threshold=0.5):
    """
    Returns:
      - predicted_label: "Benign" or "Malignant"
      - predicted_prob: probability of the predicted label (0..1)
      - malignant_prob: probability of malignant (0..1)
    """
    img = image.load_img(image_file, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # sigmoid output -> malignant probability
    pred = model.predict(img_array, verbose=0)
    malignant_prob = float(pred[0][0])

    is_malignant = malignant_prob > threshold
    predicted_label = "Malignant" if is_malignant else "Benign"

    # probability of predicted class (this is what you want to display)
    predicted_prob = malignant_prob if is_malignant else (1 - malignant_prob)

    return predicted_label, predicted_prob, malignant_prob


# streamlit UI
st.title("🩺 Skin Cancer Detection System")

st.markdown(
    "Upload a skin lesion image and the model will predict whether it is **Benign** or **Malignant**."
)

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    label, confidence, malignant_prob = predict_skin_lesion(uploaded_image, model)

    st.markdown("# 🧾 Prediction")
    st.markdown(f"### **Result: {label}**")

    # confidence bar = probability of predicted class
    st.progress(min(max(confidence, 0.0), 1.0))
    st.caption(f"Confidence: {confidence:.0%}")

    # showing probability of what it was predicted as (with the name)
    st.write(f"**{label} Probability:** {confidence:.3f}")

    # showing probability of the other class
    other_label = "Benign" if label == "Malignant" else "Malignant"
    other_probability = 1 - confidence
    st.write(f"**{other_label} Probability:** {other_probability:.3f}")

    if confidence < 0.65:
        st.warning("Low confidence prediction — try a clearer image or different lighting.")

st.markdown(
    """
### About the Model
This application uses a **2D Convolutional Neural Network (CNN)** to perform **binary classification**
of skin lesion images.

- **Input:** Skin lesion images resized to **224×224**
- **Output:** Prediction of **Benign** or **Malignant**
- **Model Output:** Sigmoid probability representing likelihood of **Malignant**
- **Decision Rule:** Predictions are thresholded at **0.5**
- **Confidence Display:** Confidence reflects the probability of the **predicted class**
- **Probability Breakdown:** Both predicted-class probability and the alternative class probability
  are shown to provide transparent model output
"""
)
