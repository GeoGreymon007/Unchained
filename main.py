import io
import os
from flask import Flask, request, jsonify
from google.cloud import vision
from google.cloud.vision_v1 import types

app = Flask(__name__)

# Set the Google Cloud Vision API credentials environment variable
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'service.json'  
client = vision.ImageAnnotatorClient()

@app.route('/detect-text', methods=['POST'])

def detect_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file:
        return jsonify({"error": "Unknown error"}), 500

    content = file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        return jsonify({"detected_text": "No text detected"}), 200

    first_text = texts[0].description
    detected_text = [text.description for text in texts]
    final_text = texts[-1].description

    return jsonify({"first_text": first_text, "detected_text": detected_text, "final_text": final_text}), 200


if __name__ == '__main__':
    app.run(debug=True)

