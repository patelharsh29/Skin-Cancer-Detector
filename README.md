# 🩺 Skin Cancer Detector

Skin cancer diagnosis from dermoscopic images is a challenging computer vision task due to variations in lighting, skin tone, lesion shape, and texture.  
This project addresses that challenge by training a **custom Convolutional Neural Network (CNN)** on thousands of labeled medical images to learn discriminative visual features indicative of malignant lesions.

An end-to-end **deep learning–based skin lesion classification system** that predicts whether an uploaded skin image is **Benign** or **Malignant**.  
The project simulates a real-world medical imaging workflow by combining **image preprocessing, a custom 7-stage CNN architecture, model evaluation, and an interactive Streamlit interface** for inference and probability-based decision support.

This system emphasizes **model transparency**, showing confidence scores and class probabilities to help users understand predictions rather than treating the model as a black box.

The trained model is integrated into a **Streamlit-based application** that allows users to upload an image and receive:
- A clear **classification result**
- A **confidence score**
- A **probability breakdown** for both classes

---

## 📊 Training & Evaluation

### Dataset Split
- **Training Images:** 11,899  
- **Testing Images:** 1,984  

### Performance Metrics

| Metric | Value |
|------|------|
| Training Accuracy | ~89.1% |
| Validation Accuracy | ~91.25% |
| Inference Time | < 0.5 seconds |
| End-to-End Prediction Time | < 500 ms |
| Test Images Evaluated | 1,984 |

---

## 🖥️ Application Features
- Image upload & preprocessing
- Real-time CNN inference
- Confidence visualization
- Probability breakdown

---

## 🛠️ Tech Stack

| Category | Tools |
|-------|------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Data Processing | NumPy |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| UI | Streamlit |

---

## ▶️ How to Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model by running:
```
skin_cancer_detection_model.ipynb
```

3. Start the application:
```bash
streamlit run main.py
```

---

## 🧠 Model Architecture — 7-Stage CNN Pipeline

1. **Conv Block 1** – Conv2D (32 filters) + MaxPooling  
2. **Conv Block 2** – Conv2D (64 filters) + MaxPooling  
3. **Conv Block 3** – Conv2D (128 filters) + MaxPooling  
4. **Flatten Layer** 
5. **Dense Layer (512 units)**  
6. **Dropout Layer**  
7. **Output Layer (Sigmoid)**  

**Total Parameters:** ~44.3M

---

## 🖼️ Application Screenshots (Pre & Post Inference)

<img width="784" height="736" alt="Screenshot 2026-01-15 at 10 58 38 PM" src="https://github.com/user-attachments/assets/0031a50d-9a5b-4b6a-8484-f0f7c9d69f22" />
<img width="409" height="885" alt="Screenshot 2026-01-15 at 10 59 46 PM" src="https://github.com/user-attachments/assets/e20cfebb-3134-48f4-9ed5-77e97c1c8c7b" />

---

## 🔮 Future Improvements
- Grad-CAM visual explainability  
- Multi-class lesion classification  
- Dataset expansion  
