from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import urllib.request
import threading
import time

app = Flask(__name__)
CORS(app)

# Model definition (same as in your notebook)
class BananaCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(BananaCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Model loading is lazy to avoid crashing the web service on startup.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = os.path.dirname(os.path.abspath(__file__))
_model_lock = threading.Lock()
_model = None
_model_error = None


def _find_weights_path():
    possible_paths = [
        os.path.join(base_dir, 'Banana', 'saved_models', 'epoch_20', 'banana_cnn_epoch_20.pth'),
        os.path.join(base_dir, 'saved_models', 'epoch_20', 'banana_cnn_epoch_20.pth'),
        os.path.join(base_dir, 'banana_cnn_epoch_20.pth')
    ]
    return next((p for p in possible_paths if os.path.exists(p)), None)


def _download_weights(target_path):
    model_url = os.environ.get(
        'MODEL_URL',
        'https://github.com/Aashish2004-mack/banana-/releases/download/v1.0/banana_cnn_epoch_20.pth'
    )
    if os.environ.get('DISABLE_MODEL_DOWNLOAD', '').lower() in ('1', 'true', 'yes'):
        return None, "Model download disabled via DISABLE_MODEL_DOWNLOAD."

    try:
        print(f"Model not found locally. Downloading from {model_url}...")
        urllib.request.urlretrieve(model_url, target_path)
        return target_path, None
    except Exception as e:
        return None, f"Failed to download model: {e}"


def get_model():
    global _model, _model_error
    if _model is not None:
        return _model, None
    if _model_error is not None:
        return None, _model_error

    with _model_lock:
        if _model is not None:
            return _model, None
        if _model_error is not None:
            return None, _model_error

        try:
            model = BananaCNN(num_classes=7).to(device)

            weights_path = _find_weights_path()
            if not weights_path:
                save_path = os.path.join(base_dir, 'banana_cnn_epoch_20.pth')
                weights_path, download_error = _download_weights(save_path)
                if download_error:
                    _model_error = download_error
                    return None, _model_error

            if weights_path and os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location=device))
                model.eval()
                _model = model
                print(f"Model loaded successfully from {weights_path}")
                return _model, None

            _model_error = "Model file not found after download attempt."
            return None, _model_error

        except Exception as e:
            _model_error = f"Could not load model weights: {e}"
            return None, _model_error

# Class names (cleaned up)
class_names = [
    'Black Sigatoka Disease',
    'Bract Mosaic Virus Disease', 
    'Healthy Leaf',
    'Insect Pest Disease',
    'Moko Disease',
    'Panama Disease',
    'Yellow Sigatoka Disease'
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.512, 0.225])
])

@app.route('/')
def index():
    # Use absolute path to find index.html
    index_path = os.path.join(base_dir, 'index.html')
    if not os.path.exists(index_path):
        return "Error: index.html not found. Please ensure index.html exists in the same directory as app.py", 404
        
    with open(index_path, 'r') as f:
        return f.read()

@app.route('/health', methods=['GET'])
def health():
    model, err = get_model()
    status = 'ok' if model is not None else 'error'
    return jsonify({'status': status, 'model_error': err})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({'status': 'active', 'message': 'Backend is running. Send a POST request with an image to classify.'})

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        model, err = get_model()
        if model is None:
            return jsonify({'error': f'Model unavailable: {err}'}), 503

        # Load and preprocess image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
        prediction = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        top3_predictions = [
            {'disease': class_names[idx.item()], 'confidence': prob.item()}
            for idx, prob in zip(top3_idx, top3_prob)
        ]
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence_score,
            'top3': top3_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use PORT environment variable if available (required for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
