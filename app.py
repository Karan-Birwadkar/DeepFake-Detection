import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import cv2

# Load the trained model
model = load_model('cnn_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    img = keras_image.load_img(image, target_size=(200, 200))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255, axis=0)
    return img_array

# Define a function to make predictions
def predict_image(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
def main():
    st.title("Image Classification: Real or Fake")
    st.write("Upload an image and we'll predict whether it's real or fake!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Make prediction when the user clicks the button
        if st.button("Predict"):
            # Process the uploaded image and make prediction
            with st.spinner('Predicting...'):
                prediction = predict_image(uploaded_file)

            # Show the prediction result
            if prediction[0][0] > 0.5:
                st.success("Prediction: Real")
            else:
                st.error("Prediction: Fake")

if __name__ == "__main__":
    main()