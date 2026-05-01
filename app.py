import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("📰 Fake News Detector")
st.caption("Enter any news text and check if it's Real or Fake")

# Input
text = st.text_area("Enter news text:", placeholder="Paste a news paragraph here...")

# Prediction
if st.button("Predict"):
    if text:
        # Clean input
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)

        # Convert text to numbers
        vec = vectorizer.transform([text])

        # Predict
        result = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        st.subheader("🧾 Result")

        if result == 1:
            st.success(f"✅ Real News")
        else:
            st.error(f"❌ Fake News")

        # Confidence
        confidence = max(prob) * 100
        st.write(f"Confidence: {confidence:.2f}%")

        # Progress bar
        st.progress(int(confidence))

    else:
        st.warning("Please enter some text")