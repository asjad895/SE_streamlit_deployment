# contentCompanion -An Ai Powered content based search engine
**euclidean_distance(vectors)**

Algorithm: This function calculates the Euclidean distance between two vectors.
Parameters:
vectors: A tuple containing two vectors, featsA and featsB.
Returns: The Euclidean distance between the two vectors.

**contrastive_loss(y, preds, margin=1)**

Algorithm: This function calculates the contrastive loss between the true labels (y) and the predicted labels (preds).
Parameters:
y: The true class label.
preds: The predicted class label.
margin: The margin parameter used in the loss calculation (default value is 1).
Returns: The computed contrastive loss.

**retrived(query, k=10)**

Algorithm: This function retrieves similar content based on a query image.
Parameters:
query: The path of the query image.
k: The number of similar content items to retrieve (default value is 10).
Returns: Three lists:
d: A list of similarity distances between the query image and retrieved images.
img: A list of paths for the retrieved images.
cap: A list of indices corresponding to the retrieved images in the training dataset.

**predict_endpoint()**

Algorithm: This function is the prediction endpoint for the Streamlit app. It retrieves the uploaded image, calls the retrived() function to get similar content, and displays the results.
Parameters: None.
Returns: None.
The remaining code is responsible for setting up the Streamlit app, including the user interface and the main execution flow. It consists of the following parts:

Importing necessary libraries and modules.
Loading the TensorFlow model and the training dataset.
Defining the prediction function for the Streamlit app.
Setting up the Streamlit app UI with file upload, number input, and search button.
Calling the prediction function when the search button is clicked.
Displaying the results in the UI.
Adding a sidebar with options for AI Presentation and Documentation.
Adding a footer at the bottom of the page.
Note: The code for running the app at the end is commented out, but you can uncomment it if you want to run the app directly without using the Streamlit command.
## Deployment:
streamlit webapp is deployed on streamlit cloud,you can access it by this link-
href="https://contentcompanion.streamlit.app/"
# Demo:
