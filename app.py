import io
import base64
import torch
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

app = Flask(__name__, static_folder='static')
CORS(app)

# ─── Load YOLOv5 model once at startup ───────────────────────────────────────
# Uses yolov5s (small, fast). Change to 'yolov5m', 'yolov5l' for better accuracy.
# If you have a custom trained model, replace with: torch.hub.load('.', 'custom', path='runs/train/exp/weights/best.pt', source='local')
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4   # confidence threshold (0.0 - 1.0)
model.iou = 0.45   # NMS IOU threshold
print("Model loaded! Starting server...")

# ─── Serve the frontend ───────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# ─── Detection endpoint ───────────────────────────────────────────────────────
@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image from frontend
        img_b64 = data['image']
        if ',' in img_b64:
            img_b64 = img_b64.split(',')[1]  # strip "data:image/jpeg;base64," prefix

        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Run YOLOv5 inference
        results = model(img, size=640)

        # Parse results into clean JSON
        df = results.pandas().xyxy[0]  # dataframe with columns: xmin, ymin, xmax, ymax, confidence, class, name

        detections = []
        for _, row in df.iterrows():
            detections.append({
                'label': row['name'],
                'confidence': round(float(row['confidence']), 3),
                'box': [
                    int(row['xmin']),
                    int(row['ymin']),
                    int(row['xmax']),
                    int(row['ymax'])
                ]
            })

        return jsonify({'detections': detections, 'count': len(detections)})

    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)