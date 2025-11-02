"""
app.py
Flask app to accept student info + photo, predict facial emotion using a pre-trained Keras model,
save the info + image blob to an SQLite DB, and show the result.

Windows-focused instructions later in the README part of the message.
"""

import os
import io
import sqlite3
from datetime import datetime

from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import numpy as np
import base64

# TensorFlow / Keras
from tensorflow.keras.models import load_model

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "static", "uploads")
DB_PATH = os.path.join(PROJECT_ROOT, "database.db")
MODEL_PATH = os.path.join(PROJECT_ROOT, "face_emotionModel.h5")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace_this_with_a_random_secret_for_production"

# --- Load Model ---
# NOTE: If your model needs a different input size or color channels, adjust preprocess_image().
print("Loading model from:", MODEL_PATH)
model = load_model(MODEL_PATH)
# If using TensorFlow with custom objects you may need to pass them in load_model(...)

# --- Emotion label & friendly messages mapping ---
# FER2013 typical order: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

FRIENDLY_MESSAGES = {
    "Angry": "You look angry. Take a breath — everything's going to be okay.",
    "Disgust": "You look displeased. Want to talk about what's bothering you?",
    "Fear": "You look frightened. Are you feeling anxious?",
    "Happy": "You look happy — nice smile! Keep it up 😊",
    "Sad": "You are frowning. Why are you sad?",
    "Surprise": "You look surprised! Something unexpected happened?",
    "Neutral": "You look calm and neutral."
}

# --- Database helpers ---
def init_db():
    """Create the database and table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            surname TEXT,
            firstname TEXT,
            matric_number TEXT,
            email TEXT,
            emotion TEXT,
            message TEXT,
            image_path TEXT,
            image_blob BLOB,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def save_to_db(surname, firstname, matric, email, emotion, message, image_path, image_bytes):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO students
        (surname, firstname, matric_number, email, emotion, message, image_path, image_blob, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (surname, firstname, matric, email, emotion, message, image_path, image_bytes, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

# --- Utilities ---
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(pil_image):
    """
    Convert a PIL image to the model input.
    This is written assuming a FER2013-like model expecting 48x48 grayscale normalized to [0,1]
    and a model input shape (1, 48, 48, 1) or (1, 48, 48).
    If your model expects color (3 channels) or a different size, change this function.
    """
    # Convert to grayscale
    img = pil_image.convert("L")
    # Resize to 48x48 (change here if your model expects a different size)
    img = ImageOps.fit(img, (48, 48), Image.ANTIALIAS)
    arr = np.asarray(img).astype("float32") / 255.0  # normalize to [0,1]
    arr = arr.reshape(1, 48, 48, 1)  # batch + height + width + channel
    return arr

def image_file_to_bytes(file_storage):
    """Return raw bytes for saving to DB."""
    file_storage.stream.seek(0)
    return file_storage.read()

def pil_to_datauri(pil_image, fmt="JPEG"):
    """Convert PIL image to data URI for embedding in HTML."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=fmt)
    buffer.seek(0)
    img_bytes = buffer.read()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{base64_str}"

# --- Routes ---
@app.route("/", methods=["GET"])
def index():
    """
    Serve the index form (templates/index.htm)
    Fields: surname, firstname, matric_number, email, file
    """
    return render_template("index.htm")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/predict", methods=["POST"])
def predict():
    # Ensure DB exists
    init_db()

    # Get form fields
    surname = request.form.get("surname", "").strip()
    firstname = request.form.get("firstname", "").strip()
    matric_number = request.form.get("matric_number", "").strip()
    email = request.form.get("email", "").strip()

    # File handling
    if "photo" not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files["photo"]
    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{surname}_{matric_number}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}")
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.seek(0)
        file.save(save_path)

        # Also read bytes to store in DB
        file.seek(0)
        image_bytes = image_file_to_bytes(file)

        # Prepare PIL image for prediction (open saved copy to ensure compatibility)
        pil_image = Image.open(save_path).convert("RGB")

        # Preprocess for model (adjust if necessary for your model)
        model_input = preprocess_image(pil_image)

        # Predict
        preds = model.predict(model_input)
        # If model outputs probabilities for classes
        if preds.ndim == 2 and preds.shape[1] >= len(EMOTION_LABELS):
            pred_idx = np.argmax(preds[0])
            emotion = EMOTION_LABELS[pred_idx]
            confidence = float(np.max(preds[0]))
        else:
            # fallback: if model outputs single value or different shape
            pred_idx = int(np.argmax(preds))
            emotion = EMOTION_LABELS[pred_idx]
            confidence = 0.0

        # Choose friendly message
        message = FRIENDLY_MESSAGES.get(emotion, f"You look {emotion}.")

        # Save to DB (we store image bytes as BLOB and also path)
        try:
            save_to_db(surname, firstname, matric_number, email, emotion, message, save_path, image_bytes)
        except Exception as e:
            # non-fatal, but inform developer
            print("DB save error:", e)

        # Convert PIL image to data URI for display
        data_uri = pil_to_datauri(pil_image, fmt="JPEG")
        return render_template("result.htm", surname=surname, firstname=firstname,
                               matric_number=matric_number, email=email,
                               emotion=emotion, confidence=confidence, message=message,
                               image_data=data_uri)
    else:
        flash("Allowed file types: png, jpg, jpeg, gif")
        return redirect(url_for("index"))

# --- Run ---
if __name__ == "__main__":
    # Initialize DB on start
    init_db()
    # For development only: set debug=True on Windows if desired
    app.run(host="0.0.0.0", port=5000, debug=True)
