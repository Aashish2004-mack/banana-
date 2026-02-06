from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import urllib.request

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

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BananaCNN(num_classes=7).to(device)

# Load trained weights (adjust path as needed)
# Use absolute path relative to this file to ensure it works on Render
base_dir = os.path.dirname(os.path.abspath(__file__))

try:
    # Check multiple possible locations for the model
    possible_paths = [
        os.path.join(base_dir, 'Banana', 'saved_models', 'epoch_20', 'banana_cnn_epoch_20.pth'),
        os.path.join(base_dir, 'saved_models', 'epoch_20', 'banana_cnn_epoch_20.pth'),
        os.path.join(base_dir, 'banana_cnn_epoch_20.pth')
    ]
    
    weights_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    # If model is missing, try to download it from GitHub Releases
    if not weights_path:
        # This URL matches the release instructions in Step 1
        MODEL_URL = "https://github.com/Aashish2004-mack/banana-/releases/download/v1.0/banana_cnn_epoch_20.pth"
        
        print(f"Model not found locally. Downloading from {MODEL_URL}...")
        try:
            save_path = os.path.join(base_dir, 'banana_cnn_epoch_20.pth')
            urllib.request.urlretrieve(MODEL_URL, save_path)
            weights_path = save_path
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download model: {e}")

    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully from {weights_path}")
    else:
        print("Warning: Model file not found and could not be downloaded. Using untrained model.")

except Exception as e:
    print(f"Warning: Could not load model weights: {e}. Using untrained model.")

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

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
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