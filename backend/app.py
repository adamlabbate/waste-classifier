import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tempfile
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download
from src.model import WasteClassifier
from src.predict import predict, device

MODEL_PATH = hf_hub_download(repo_id="adamlabbate/waste-classifier", filename="waste_classifier.pth")

app = Flask(__name__)
CORS(app)

model = WasteClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    suffix = os.path.splitext(file.filename)[1] or '.jpg'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = predict(tmp_path, model, device)
        return jsonify({'class': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
