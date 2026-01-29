import streamlit as st
import numpy as np
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
# ===== Load model & preprocessor =====
@st.cache_resource
def load_artifacts():
    model = load_model("emotion_detection_model.h5")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_artifacts()

tokenizer = preprocessor["tokenizer"]
label_encoder = preprocessor["label_encoder"]
max_len = preprocessor["max_len"]

# ===== UI =====
st.title("😊 Emotion Detection from Text")

content = st.text_area(
    "Content",
    placeholder="Enter text to detect sentiment/emotion"
)

# ===== Prediction =====
if st.button("Predict Sentiment"):
    if content.strip() == "":
        st.warning("Please enter text in the content field.")
    else:
        with st.spinner("Predicting..."):
            sequence = tokenizer.texts_to_sequences([content])
            padded = pad_sequences(sequence, maxlen=max_len)

            prediction = model.predict(padded)
            predicted_index = np.argmax(prediction)
            sentiment = label_encoder.inverse_transform([predicted_index])[0]
            confidence = np.max(prediction)

        st.success(f"**Predicted Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")
