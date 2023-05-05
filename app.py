import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the TensorFlow model
model = tf.keras.models.load_model('path/to/your/model.h5')

# Define a prediction function
def predict(image):
    # Preprocess the image
    img = Image.open(image)
    img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))[0]

    # Return the prediction as a list
    return pred.tolist()

# Set up the Streamlit app
st.title("TensorFlow Model API")

# Define the prediction endpoint
@st.experimental_memo
def predict_endpoint():
    # Get the uploaded file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Get the prediction
        prediction = predict(uploaded_file)

        # Show the prediction
        st.write(prediction)

# Run the app
if __name__ == "__main__":
    predict_endpoint()
