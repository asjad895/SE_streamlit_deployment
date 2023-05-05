import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
from keras.utils import load_img,img_to_array
import keras.backend as K
def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors

    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)

    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def contrastive_loss(y, preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, preds.dtype)

    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)

    # return the computed contrastive loss to the calling function
    return loss
# Load the TensorFlow model
em_model = tf.keras.models.load_model('models/embedding_model.h5',custom_objects={'contrastive_loss': contrastive_loss})
train=pd.read_csv('train.csv')

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Define a prediction function
def retrived(query):
    # get the indices that would sort the distances array in ascending order
    img = load_img(query, color_mode='rgb', target_size=(224, 224))
    img = img_to_array(img)
    query_img = img / 255.0
    query_emb = em_model.predict(np.expand_dims(query_img, axis=0))[0]

# Compute the distances between the query embedding and all the retrieval set embeddings
    retrieval_set_embeddings = np.load("models/retrieval_set_embeddings.npz")
    retrieval_set_embs = retrieval_set_embeddings["embeddings"]
    retrieval_set_filenames = retrieval_set_embeddings["filenames"]
    distances = [euclidean_distance(query_emb, emb) for emb in retrieval_set_embs]
    sorted_indexes = np.argsort(distances)

# print the top-k images with their corresponding distances
    k = 10
    d=[]
    img=[]
    cap=[]
    for i in range(k):
        image_index = sorted_indexes[i]
        image_path=retrieval_set_filenames[image_index]
        p=image_path.split('model_images/')[1]
        # filter the rows containing the path
        filtered_df = train[train["path"] == p]

# get the indices of the filtered rows
        indices = filtered_df.index.tolist()
        cap.append(indices)
        image_distance = distances[image_index]
        d.append(image_distance)
        img.append(image_path)
        st.table(d)
        st.table(img)
        st.table(cap)
    return d,img,cap

# Set up the Streamlit app
st.title("TensorFlow Model API")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# Define the prediction endpoint
@st.cache_data
def predict_endpoint():
    # Get the uploaded file

    if uploaded_file is not None:
        # Get the prediction
        prediction = retrived(uploaded_file)

        # Show the prediction
        st.write(prediction)

# Run the app
if __name__ == "__main__":
    predict_endpoint()
