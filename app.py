import streamlit as st
import pickle
import numpy as np
import re
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load model and vectorizer
try:
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error(f"❌ Failed to load model/vectorizer: {e}")
    st.stop()

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text.lower()

# Streamlit app configuration
st.set_page_config(page_title="🍽️ Restaurant Review Sentiment Analyzer", layout="centered")
st.title("🍽️ Restaurant Review Sentiment Analyzer")
st.markdown("Enter a restaurant review below to detect sentiment using a machine learning model.")

# User input
review = st.text_area("✍️ Your Review", height=150)

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a valid review.")
    else:
        # Preprocess & transform
        cleaned_review = preprocess_text(review)
        vectorized_input = vectorizer.transform([cleaned_review]).toarray()

        # Model prediction
        pred = model.predict(vectorized_input)[0]
        pred_proba = model.predict_proba(vectorized_input)[0]
        confidence = round(np.max(pred_proba) * 100, 2)

        # TextBlob fallback sentiment (polarity: -1 to +1)
        blob_score = TextBlob(review).sentiment.polarity

        # Sentiment adjustment with TextBlob
        if -0.2 <= blob_score <= 0.2:
            pred = 2  # Neutral
        elif pred == 1 and blob_score < -0.2:
            pred = 0  # Adjust to Negative
        elif pred == 0 and blob_score > 0.3:
            pred = 1  # Adjust to Positive

        # Display prediction
        if pred == 1:
            st.success("✅ **Positive Review 😊**")
            st.markdown("🎉 Great! Keep up the good work.")
        elif pred == 2:
            st.info("😐 **Neutral Review 😐**")
            st.markdown("📌 Try to gather more detailed feedback.")
        else:
            st.error("❌ **Negative Review 😞**")
            st.markdown("💡 **Recommendations & Feedback**")
            st.write("🔎 Suggestion: Improve quality or service.")
            st.write("⚠️ Negative feedback highlights issues to fix.")

        # Confidence Progress
        st.subheader("📊 Model Confidence")
        st.progress(min(confidence / 100, 1.0))
        st.write(f"🧠 **Confidence:** `{confidence:.2f}%`")

        # Sentiment Distribution Pie Chart
        st.subheader("📈 Sentiment Distribution (Demo)")
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [1 if pred == 1 else 0, 1 if pred == 2 else 0, 1 if pred == 0 else 0]
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("🚀 Made with ❤️ by **Saksham Sharma** for IBM SkillsBuild Internship")
