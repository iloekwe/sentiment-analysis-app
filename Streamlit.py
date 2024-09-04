import joblib
import streamlit as st
from PIL import Image

# Load the pre-trained sentiment analysis model, CountVectorizer, and LabelEncoder using joblib
model = joblib.load('sgd_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load and display an image (optional)
image = Image.open('Sentiment.jpg')

# Display image centered
st.image(image, width=400, caption='Sentiment Analysis')  # Adjust the width as needed

# Centralize the title of the web application
st.markdown("<h1 style='text-align: center;'>Customer Review Sentiment Analysis</h1>", unsafe_allow_html=True)

# Sidebar for user input with an encouraging message
st.sidebar.markdown(
    """
    <h2 style='text-align: center;'>Welcome!</h2>
    <p style='text-align: center;'>We can predict the sentiment of your customer reviews using our machine learning model. Please enter your review below and click 'Analyze Sentiment' to get started!</p>
    """, unsafe_allow_html=True
)

def get_user_input():
    st.sidebar.header("Input from User")
    st.sidebar.subheader("Enter the customer review text below:")
    review_text = st.sidebar.text_area("Customer Review", "Type your review here...")
    return review_text

# Get user input
review_text = get_user_input()

# Display the original review text
st.write(f"Input Review: {review_text}")

# Predict sentiment
if st.sidebar.button("Analyze Sentiment"):
    # Transform the review_text using the loaded CountVectorizer
    review_vectorized = vectorizer.transform([review_text])

    # Predict sentiment using the model
    prediction = model.predict(review_vectorized)

    # Map prediction to sentiment
    sentiment = label_encoder.inverse_transform(prediction)[0]

    # Display the result
    st.header("Sentiment Prediction")
    st.write(f"The sentiment of the review is: **{sentiment}**")
