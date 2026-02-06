# Banana Disease Classification

A deep learning web application for classifying banana leaf diseases using PyTorch and Flask.

## Features

- **7 Disease Classes**: Black Sigatoka, Bract Mosaic Virus, Healthy Leaf, Insect Pest, Moko, Panama, Yellow Sigatoka
- **Web Interface**: Simple drag-and-drop image upload
- **Real-time Prediction**: Instant classification with confidence scores
- **Top 3 Predictions**: Shows multiple possible diagnoses

## Project Structure

```
Banana/
├── app.py                 # Flask backend server
├── upload.html           # Frontend web interface
├── requirements.txt      # Python dependencies
├── run_server.bat       # Windows batch script to run server
├── .gitignore           # Git ignore file
├── Banana/
│   ├── data_preprocessing.ipynb  # Model training notebook
│   ├── data/            # Training and test datasets
│   │   ├── train/       # Training images by disease class
│   │   └── test/        # Test images by disease class
│   ├── saved_models/    # Trained model weights
│   │   ├── epoch_01/    # Model weights from epoch 1
│   │   ├── epoch_02/    # Model weights from epoch 2
│   │   ├── ...
│   │   └── epoch_20/    # Final model weights
│   │       └── banana_cnn_epoch_20.pth
│   ├── saved_models_224/ # Alternative model weights (224x224)
│   └── prediction/      # Sample prediction images
└── README.md            # Project documentation
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aashish2004-mack/banana-.git
   cd banana-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```
   Or on Windows: double-click `run_server.bat`

4. **Open browser** and go to `http://localhost:5000`

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for hosting options:
- **Render** (Recommended - Free)
- **Railway** (Easy deployment)
- **Hugging Face Spaces** (Best for ML)
- **ngrok** (Quick local sharing)

**Note**: GitHub Pages doesn't support Python backends. Use the deployment options above.

## Usage

1. Open the web application in your browser
2. Click the upload area or drag and drop a banana leaf image
3. Click "Classify Image" to get the prediction
4. View the results with confidence scores

## Model Architecture

- **Custom CNN**: 3-block convolutional neural network
- **Input Size**: 512x512 RGB images
- **Training**: 20 epochs with data augmentation
- **Accuracy**: ~85% on test set

## Technologies Used

- **Backend**: Flask, PyTorch, torchvision
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: PIL (Pillow)
- **Model**: Custom CNN architecture

## Dataset

The model is trained on augmented banana leaf images with 7 disease categories:
- Augmented Banana Black Sigatoka Disease
- Augmented Banana Bract Mosaic Virus Disease
- Augmented Banana Healthy Leaf
- Augmented Banana Insect Pest Disease
- Augmented Banana Moko Disease
- Augmented Banana Panama Disease
- Augmented Banana Yellow Sigatoka Disease

## API Endpoints

- `GET /` - Serves the web interface
- `POST /predict` - Accepts image upload and returns prediction

### Prediction Response Format
```json
{
  "prediction": "Healthy Leaf",
  "confidence": 0.95,
  "top3": [
    {"disease": "Healthy Leaf", "confidence": 0.95},
    {"disease": "Black Sigatoka Disease", "confidence": 0.03},
    {"disease": "Yellow Sigatoka Disease", "confidence": 0.02}
  ]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.