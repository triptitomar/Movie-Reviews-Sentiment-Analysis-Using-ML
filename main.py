import streamlit as st
import pickle

# Load trained sentiment analysis model
with open("sentiment_modell.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 🎨 Customizing the UI
st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬", layout="wide")

# Adding a background style with new colors
st.markdown(
    """
    <style>
        body {
            background-color: #F8F9F9; /* Light Grey */
        }
        .main {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 20px rgba(0,0,0,0.1);
        }
        h1 {
            color: #FF5733;
            text-align: center;
        }
        .stTextArea textarea {
            font-size: 16px;
            padding: 10px;
        }
        .stButton button {
            background-color: #145A32 !important;
            color: white !important;
            font-size: 18px;
            padding: 10px 20px;
        }
        /* Sidebar Background */
        section[data-testid="stSidebar"] {
            background-color: #B22222 !important; /* Light Blue */
            padding: 20px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🎭 Sidebar
st.sidebar.title("ℹ️ About the App")
st.sidebar.write(
    """
    - This app analyzes the sentiment of movie reviews.  
    - Enter a review and click **Analyze Sentiment** to see if it's Positive or Negative.  
    - Uses **TF-IDF Vectorization** & Machine Learning Model for prediction.  
    """
)

st.sidebar.subheader("🔍 How It Works")
st.sidebar.write(
    """
    1️⃣ Type or paste a movie review.  
    2️⃣ Click on **Analyze Sentiment**.  
    3️⃣ Get a sentiment prediction with confidence percentage.  
    """
)

# 🎬 Main Title
st.markdown("<h1>🎬 Movie Review Sentiment Analysis</h1>", unsafe_allow_html=True)

# User Input
review = st.text_area("✍️ Enter your movie review:", height=150, placeholder="Type your review here...")

# Analyze Button
if st.button("🎭 Analyze Sentiment"):
    if review.strip():  # Ensure input is not empty
        # Vectorize input
        review_vectorized = vectorizer.transform([review])

        # Predict sentiment
        try:
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(review_vectorized.toarray())[0]
                positive_prob = probabilities[1] * 100
                negative_prob = probabilities[0] * 100

                sentiment = "😊 Positive" if positive_prob > negative_prob else "☹️ Negative"

                # Display results with color highlighting
                st.markdown(f"<h3 style='color:green;'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
                st.progress(int(positive_prob) if positive_prob > negative_prob else int(negative_prob))
                st.write(f"**Confidence:** 🎯 {positive_prob:.2f}% Positive | {negative_prob:.2f}% Negative")

            else:
                prediction = model.predict(review_vectorized.toarray())
                sentiment = "😊 Positive" if prediction[0] == 1 else "☹️ Negative"
                st.markdown(f"<h3 style='color:green;'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

    else:
        st.warning("⚠️ Please enter a review before analyzing!")
