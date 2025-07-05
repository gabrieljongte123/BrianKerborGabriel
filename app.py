from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = None

def load_model():
    global model
    if os.path.exists("mnist_model.h5"):
        model = tf.keras.models.load_model("mnist_model.h5")
        print("Model loaded successfully!")
    else:
        print("Model file not found. Please run model_train.py first.")

def preprocess_image(image_data):
    """Preprocess the canvas image for model prediction"""
    try:
        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to grayscale and resize to 28x28
        img = img.convert('L').resize((28, 28))
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype('float32') / 255.0
        
        # Invert the image (MNIST digits are white on black background)
        img_array = 1.0 - img_array
        
        # Reshape for model input (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Preprocess the image
        processed_image = preprocess_image(data['image'])
        if processed_image is None:
            return jsonify({"error": "Failed to process image"}), 400
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Get all probabilities for visualization
        all_probabilities = predictions[0].tolist()
        
        return jsonify({
            "prediction": predicted_digit,
            "confidence": confidence,
            "probabilities": all_probabilities
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5001) 