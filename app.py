from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('finalmodel.keras')

# Define the class mappings (adjust to your actual mappings)
class_mappings = {0: 'Glioma', 1: 'Meningioma', 2: 'NoTumor', 3: 'Pituitary'}

# Function to preprocess the uploaded image
def preprocess_image(img_path, target_size=(168, 168)):
    img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0  # Normalize the image to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension: (1, height, width, 1)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']
        if file:
            # Save the file to the uploads folder
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Preprocess the image
            img_array = preprocess_image(file_path)

            # Make a prediction
            predictions = model.predict(img_array)

            # Get the predicted class and confidence
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_label = class_mappings[predicted_class_index]
            confidence = round(np.max(predictions) * 100, 2)

            return render_template('index.html', prediction=f'{predicted_class_label} ({confidence}%)')

    return render_template('index.html', prediction="No file uploaded.")

if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
