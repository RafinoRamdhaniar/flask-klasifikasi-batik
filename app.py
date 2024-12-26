from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
from PIL import Image


app = Flask(__name__)

# load model for prediction
model_paths = {
    "mobilenetv2": "Yona_MobileNet.h5",
    "densenet121": "Yona_DenseNet121.h5",
    "xception": "Yona_Xception.h5"
}

models = {name: tf.keras.models.load_model(path) for name, path in model_paths.items()}

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/cnn", methods=['GET', 'POST'])
def cnn():
	return render_template("cnn.html")

@app.route("/classification", methods = ['GET', 'POST'])
def classification():
	return render_template("classifications.html")

# @app.route("/login", methods = ['GET', 'POST'])
# def login():
# 	return render_template("page-login.html")

@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'No file or model selected'}), 400

    file = request.files['file']
    selected_model = request.form['model']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load the selected model
        model = models.get(selected_model)
        if model is None:
            return jsonify({'error': 'Invalid model selected'}), 400

        # Preprocess the image
        img = Image.open(filepath).resize((224, 224))  # Adjust size based on your model
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]

        # Define class labels
        class_labels = {
            0: "Batik Lampung",
            1: "Batik Yogyakarta",
            2: "Bukan Batik"
        }

        predicted_label = class_labels.get(predicted_class, "Unknown")
        os.remove(filepath)  # Remove the uploaded file after prediction

        return jsonify({
            'predicted_class': predicted_label,
            'confidence': f"{confidence:.2f}"
        })
    else:
        return jsonify({'error': 'Invalid file format'}), 400


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)