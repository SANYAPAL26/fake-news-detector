import os

if not os.path.exists("model.pkl"):
    import train_model
import streamlit as st
import pickle
import re

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")

text = st.text_area("Enter news text:")

if st.button("Predict"):
    if text:
        # Clean input
        text = text.lower()
        import re
        text = re.sub(r"[^\w\s]", "", text)

        vec = vectorizer.transform([text])
        result = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        st.subheader("🧾 Result")

        if result == 1:
            st.success(f"✅ Real News")
        else:
            st.error(f"❌ Fake News")

        # Show confidence
        confidence = max(prob) * 100
        st.write(f"Confidence: {confidence:.2f}%")

        # Progress bar
        st.progress(int(confidence))

    else:
        st.warning("Please enter text")