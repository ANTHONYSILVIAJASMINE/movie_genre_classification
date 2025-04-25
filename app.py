import streamlit as st
import joblib
import numpy as np

# Load model, vectorizer, and label binarizer
@st.cache_resource
def load_components():
    model = joblib.load("genre_classifier.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    mlb = joblib.load("mlb.pkl")
    return model, vectorizer, mlb

# Streamlit app
def main():
    st.set_page_config(page_title="Movie Genre Classifier", layout="centered")
    st.title("🎬 Movie Genre Classifier")
    st.write("Enter a movie description and get predicted genres!")

    # Load components
    try:
        model, vectorizer, mlb = load_components()
    except Exception as e:
        st.error(f"Failed to load model or vectorizer. Error: {e}")
        return

    # Debug print (optional)
    # st.write("✅ Model and vectorizer loaded.")

    # Input
    overview = st.text_area("Movie Overview", height=200, placeholder="e.g., A young boy discovers he has magical powers and attends a wizarding school.")

    if st.button("Predict Genres"):
        if not overview.strip():
            st.warning("Please enter a movie overview.")
        else:
            # Transform input
            overview_vectorized = vectorizer.transform([overview])
            prediction = model.predict(overview_vectorized)[0]

            # Convert prediction to genre labels
            predicted_genres = [genre for genre, is_present in zip(mlb.classes_, prediction) if is_present]

            if predicted_genres:
                st.success("🎯 Predicted Genres:")
                st.write(", ".join(predicted_genres))
            else:
                st.info("No genres predicted. Try a more descriptive overview.")

if __name__ == "__main__":
    main()
