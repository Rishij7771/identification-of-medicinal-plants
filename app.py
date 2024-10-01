from flask import Flask, request, render_template, redirect, url_for
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('svm_model.pkl', 'rb') as file:
    best_estimator = pickle.load(file)

categories = ['Tulsi', 'Sandalwood', 'Rose_apple', 'Rasna', 'Pomegranate', 'Peepal', 'Parijata', 'Oleander',
             'Neem', 'Mint', 'Mexican_Mint', 'Mango', 'lemon', 'Karanda', 'Jasmine', 'Jamun', 'Jamaica_Cherry-Gasagase', 
             'Jackfruit', 'Indian_Mustard', 'Indian_Beech', 'Hibiscus', 'Guava', 'Fenugreek', 'Drumstick', 'Curry',
             'Crape_jasmine', 'Betel', 'Basale', 'Arive-Dantu', 'Roxburgh_fig']

# Define function for prediction
def image_classification_prediction(image_path):
    img = imread(image_path)
    img_resized = resize(img, (15, 15))
    img_flatten = img_resized.flatten()
    img_array = np.asarray(img_flatten)
    result = best_estimator.predict(img_array.reshape(1, -1))
    predicted_category = categories[result[0]]
    return predicted_category

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure your HTML file is named 'index.html'

# Route to handle the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if file is uploaded
        if 'myfile' not in request.files:
            return redirect(request.url)
        
        file = request.files['myfile']
        
        if file.filename == '':
            return redirect(request.url)
        
        # Save the uploaded file to the uploads directory
        upload_dir = 'static/uploads'  # Store files in the static directory
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        # Call the prediction function with the saved image
        predicted_category = image_classification_prediction(file_path)

        # Render result.html and pass the prediction result and filename
        return render_template('result.html', prediction=predicted_category, filename=file.filename)

if __name__ == "__main__":
    app.run(debug=True)
