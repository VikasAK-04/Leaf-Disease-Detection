from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
from PIL import Image, UnidentifiedImageError
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json
import os

# ============================================================
# 🌿 LEAF DISEASE DETECTION FLASK API
# ============================================================
app = Flask(__name__)
CORS(app)  # ✅ Enables cross-origin access for frontend/Spring Boot

# ------------------------------------------------------------
# ✅ 1. LOAD TRAINED MODEL
# ------------------------------------------------------------
print("🔄 Loading trained model...")
try:
    model = keras.models.load_model("model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

# ------------------------------------------------------------
# ✅ 2. LOAD CLASS LABELS
# ------------------------------------------------------------
if os.path.exists('class_indices.json'):
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    CLASS_NAMES = {v: k for k, v in class_indices.items()}
    CLASS_NAMES = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    print(f"✅ Loaded {len(CLASS_NAMES)} class names: {CLASS_NAMES}")
else:
    CLASS_NAMES = [
        "Tomato___Bacterial_spot",
        "Tomato___Healthy",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold"
    ]
    print(f"⚠️ Defaulting to class names: {CLASS_NAMES}")

# ------------------------------------------------------------
# ✅ 3. HOME ENDPOINT
# ------------------------------------------------------------
@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "message": "🌿 Tomato Leaf Disease Detection API is live!",
        "available_classes": CLASS_NAMES,
        "usage": {
            "POST /predict": "Upload an image using key 'file'"
        }
    })

# ------------------------------------------------------------
# ✅ 4. PREDICTION ENDPOINT
# ------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ----------------------------------------------------
        # 🧾 Validate Request
        # ----------------------------------------------------
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "File missing",
                "message": "Please upload an image using the 'file' key."
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "Empty filename",
                "message": "Please select a valid image file."
            }), 400

        # ----------------------------------------------------
        # 🖼️ Load and Preprocess Image
        # ----------------------------------------------------
        try:
            img = Image.open(file.stream).convert("RGB")
        except UnidentifiedImageError:
            return jsonify({
                "success": False,
                "error": "Invalid image format",
                "message": "Uploaded file is not a valid image."
            }), 400

        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # ----------------------------------------------------
        # 🔮 Model Prediction
        # ----------------------------------------------------
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded",
                "message": "Please check if the model.h5 file exists and is accessible."
            }), 500

        predictions = model.predict(img_array, verbose=0)[0]
        predicted_index = np.argmax(predictions)
        confidence = float(predictions[predicted_index])

        # ----------------------------------------------------
        # 🚨 Confidence Filtering for Unknown Images
        # ----------------------------------------------------
        CONFIDENCE_THRESHOLD = 0.75  # adjustable threshold
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_label = "Unknown or Unrelated Image"
            msg = "Low confidence — uploaded image may not match known tomato leaf diseases."
            print(f"⚠️ Low Confidence ({confidence:.2f}) → Marked as Unknown.")
        else:
            predicted_label = CLASS_NAMES[predicted_index]
            msg = f"Prediction: {predicted_label} ({confidence*100:.2f}%)"
            print(f"✅ {msg}")

        # ----------------------------------------------------
        # 📊 Prepare Full Prediction Breakdown
        # ----------------------------------------------------
        all_predictions = {
            CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        sorted_predictions = dict(
            sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        )

        # ----------------------------------------------------
        # 📦 Prepare Response
        # ----------------------------------------------------
        response = {
            "success": True,
            "predicted_class": predicted_label,
            "confidence": round(confidence * 100, 2),
            "message": msg,
            "all_predictions": sorted_predictions
        }

        return jsonify(response)

    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Internal server error during prediction."
        }), 500

# ------------------------------------------------------------
# ✅ 5. HEALTH CHECK
# ------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy" if model else "error",
        "model_loaded": model is not None,
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES
    })

# ------------------------------------------------------------
# 🚀 6. START SERVER
# ------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Starting Flask ML API for Leaf Disease Detection")
    print("=" * 60)
    print("📍 Home:        http://127.0.0.1:5000/")
    print("📍 Predict:     http://127.0.0.1:5000/predict")
    print("📍 Health:      http://127.0.0.1:5000/health")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
