import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="cancer_detection_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set up Streamlit app
st.title("Cancer Detection from Tissue Images")
st.write("Upload an image to check whether it is cancer-affected or not.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    # Ensure the image has 3 channels (convert grayscale images to RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def predict(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

if uploaded_file is not None:
    try:
        # Read the image
        image = Image.open(uploaded_file)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Predict
        prediction = predict(preprocessed_image)

        # Display the image and prediction
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
        if prediction[0][0] < 0.5:
            st.write("The image is predicted to be **cancer-affected**.")
        else:
            st.write("The image is predicted to be **not cancer-affected**.")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
