# 🎯 Digit Recognition App

A web application that recognizes hand-drawn digits (0-9) using a Convolutional Neural Network trained on the MNIST dataset.

## ✨ Features

- **Interactive Drawing Canvas**: Draw digits using mouse or touch
- **Real-time Prediction**: Get instant predictions with confidence scores
- **Probability Visualization**: See confidence levels for all 10 digits
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Clean, intuitive interface with smooth animations

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Python Flask API
- **Machine Learning**: TensorFlow/Keras with CNN model
- **Dataset**: MNIST (Modified National Institute of Standards and Technology)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DigitRecognition
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (First time only)
   ```bash
   python model_train.py
   ```
   This will download the MNIST dataset and train a CNN model (~5-10 minutes).

4. **Start the Flask server**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
DigitRecognition/
├── app.py              # Flask backend API
├── model_train.py      # Model training script
├── mnist_model.h5      # Trained model (generated after training)
├── index.html          # Main HTML page
├── style.css           # CSS styles
├── script.js           # Frontend JavaScript
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🧠 Model Architecture

The CNN model consists of:
- **Input Layer**: 28x28x1 grayscale images
- **Convolutional Layers**: 3 Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling2D for dimensionality reduction
- **Dense Layers**: Fully connected layers with dropout
- **Output Layer**: Softmax activation for 10 digit classes

## 🔧 API Endpoints

- `GET /` - Serve the main HTML page
- `POST /predict` - Predict digit from canvas image
- `GET /health` - Health check endpoint

### Prediction Request Format
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

### Prediction Response Format
```json
{
  "prediction": 5,
  "confidence": 0.95,
  "probabilities": [0.01, 0.02, 0.01, 0.01, 0.01, 0.95, 0.01, 0.01, 0.01, 0.01]
}
```

## 🎨 Usage

1. **Draw a digit**: Use your mouse or finger to draw a digit (0-9) on the canvas
2. **Click Predict**: Press the "🔮 Predict Digit" button
3. **View Results**: See the predicted digit and confidence score
4. **Clear Canvas**: Use "🗑️ Clear Canvas" to start over

## 📱 Mobile Support

The app is fully responsive and supports touch drawing on mobile devices.

## 🔮 Model Performance

- **Training Accuracy**: ~99% on MNIST test set
- **Inference Time**: <100ms per prediction
- **Model Size**: ~2MB

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
1. **Backend**: Deploy to Render, Railway, or Heroku
2. **Frontend**: Deploy to Vercel, Netlify, or GitHub Pages
3. **Model**: Include `mnist_model.h5` in your deployment

### Environment Variables
- `FLASK_ENV`: Set to `production` for production
- `PORT`: Server port (default: 5000)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- TensorFlow/Keras for the ML framework
- Flask for the web framework

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**
   - Run `python model_train.py` first
   - Ensure `mnist_model.h5` exists in the project directory

2. **Import errors**
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **Canvas not drawing**
   - Check browser console for JavaScript errors
   - Ensure JavaScript is enabled

4. **Prediction fails**
   - Check Flask server logs
   - Ensure the model file is accessible
   - Verify the image preprocessing pipeline

### Performance Tips

- Use a GPU for faster model training
- Optimize image preprocessing for better accuracy
- Consider model quantization for deployment 