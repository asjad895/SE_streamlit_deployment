import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
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
train=pd.read_csv('Data/preprocessed/train.csv')

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Define a prediction function
def retrived(query,k=10):
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
    return d,img,cap
import io
st.set_page_config(
    page_title="ContentCompanion",
    page_icon="👋",
)

# Set up the Streamlit app
st.title("ContentCompanion-An AI Powered Content based Search Engine")
st.markdown(
    f"<span style='color: rgb(91, 166, 236);'>Choose an image...📩</span>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(f"👇", type=["jpg", "jpeg", "png"])
n=5
st.markdown(
    f"<span style='color: rgb(91, 166, 236);'>Choose How many content you want to see! 5, 10, 15... 🕠</span>",
    unsafe_allow_html=True
)
n = st.number_input("👇",
    min_value=5,
    max_value=20
)



st.info(f"No of queried Image_{n}:",icon="🤖")

if uploaded_file is not None:
    if hasattr(uploaded_file, 'read'):
        st.write(uploaded_file)
        image = Image.open(uploaded_file)
        st.image(image, width=224, caption='A Query Image')
    else:
        st.error("Invalid file")

submit_button = st.button("Search")

# Define the prediction endpoint
# @st.cache_data
def predict_endpoint():
    # Get the uploaded file
    if uploaded_file is not None:
        # Get the prediction
        d,img,cap = retrived(uploaded_file,k=n)
        imges=[]
        py_list = [item[0] for item in cap]
        filtered_df = train[train.index.isin(py_list)]
        cap=filtered_df['caption']
        for i in range(len(img)):
            p=img[i].split('model_images/')[1]
            imges.append(p)
        # Show the prediction
        df = pd.DataFrame({'Path': imges, 'Caption': cap, 'Similarity distance': d})
        st.header('Results-Retrieved Content by our Model')
        st.dataframe(df)
        # Define the number of images to display in a row
        num_images_per_row = 3
        col1, col2, col3 = st.columns(num_images_per_row)
# Loop over the rows of the dataframe and display the images
        for i, row in df.iterrows():
            file_name = row["Path"]
            file_name=str("Data/images/")+file_name
            cap=row["Caption"]
            sim_distance = row["Similarity distance"]
            # Create the caption for the image
            caption = f"{cap}_{sim_distance:.4f}"
            # Display the image
            if i % num_images_per_row == 0:
                col1, col2, col3= st.columns(num_images_per_row)
                with col1:
                    st.image(file_name, use_column_width=True, caption=caption)
            if i % num_images_per_row == 1:
                with col2:
                    st.image(file_name, use_column_width=True, caption=caption)
            if i % num_images_per_row == 2:
                with col3:
                    st.image(file_name, use_column_width=True, caption=caption)
            # if i % num_images_per_row == 3:
            #     with col4:
            #         st.image(file_name, width=150, caption=caption)
            # if i % num_images_per_row == 4:
            #     with col5:
            #         st.image(file_name, width=150,caption=caption)
    st.success("This is the content our model queried based on available 700 images vector and current trained network!enjoyed🙌")

# If the button is clicked and a file is uploaded, call the predict_endpoint function
if submit_button and uploaded_file is not None:
    with st.spinner("Wait for Results....."):
        predict_endpoint()

else:
    if submit_button:
        with st.spinner("Wait for Results....."):
            st.warning("Upload your Query Content Before Search👍")

page=["AI Presentation","Documenation"]
with st.sidebar:
    op=st.selectbox(label="Select A Page:",options=page)
    if op=="AI Presentation":
        st.info("This is the Presenation for Ai Hackathon on Ai Planet.")
        st.markdown("[Ai Presentation](https://asjad895.github.io/SE_presentation/)")
    if op=="Documenation":
        st.info("This is Documenation for this web app and Notebook/Model.")
        st.markdown("[Documentation](https://github.com/asjad895/SE_streamlit_deployment/blob/main/README.md)")


# Add a footer at the bottom of the page
footer_html = """

<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    padding: 10px 0;
    text-align: center;
    height: 100px; /* Height of the footer */
    background-color: #110909;
}
</style>
<div>
    <footer class="footer">
        <section class="mb-4">
          <a 
            class="btn text-white btn-floating m-5"
            style="background-color: #0481c4;"
            href="https://www.linkedin.com/in/md-asjad-314501211/"role="button">
            <i class="fab fa-linkedin-in me-2"></i>
            Linkedin
          </a>
          <!-- Github -->
          <a
            class="btn text-white btn-floating m-5"
            style="background-color: #000000;"
            href="https://github.com/asjad895"role="button">
            <i class="fab fa-github me-2"></i>
            Github
          </a>
      <!-- Copyright -->
      <div class="text-center p-3" style="background-color: rgba(23, 3, 102, 0.2);">
        © 2023 Copyright:
        <a class="text-white" href="#">Evolve.ai.com</a>
      </div>
      <!-- Copyright -->
    </footer>
  </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
# Run the app
# if __name__ == "__main__":
#     predict_endpoint()
